# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.model import str_dtype_to_torch_dtype
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    get_draft_quant_config,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import IQuestQ1Attention, IQuestQ1MoEBlock, IQuestQ1RMSNorm

logger = init_logger(__name__)


def validate_recursive_draft(speculative_config) -> None:
    """Validate the recursive checkpoint after vLLM's EAGLE config wrapping."""
    if speculative_config.method != "eagle":
        raise ValueError(
            "IQuestQ1 recursive MTP requires method='eagle' in the model plugin"
        )
    if speculative_config.parallel_drafting:
        raise ValueError("IQuestQ1 recursive MTP requires serial drafting")
    draft = speculative_config.draft_model_config.hf_config
    original = getattr(draft, "model", draft)
    target = speculative_config.target_model_config.hf_config
    if original.model_type != "iquest_q1_mtp" or target.model_type != "iquest_q1":
        raise ValueError(
            "Recursive MTP requires an IQuestQ1 target and iquest_q1_mtp checkpoint"
        )
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError(
            "Recursive draft hidden size and vocabulary must match the target"
        )
    if speculative_config.num_speculative_tokens > draft.num_draft_slots:
        logger.warning(
            "Recursive MTP depth exceeds the checkpoint's trained num_draft_slots=%d",
            draft.num_draft_slots,
        )


class IQuestQ1RecursiveInnerLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.self_attn = IQuestQ1Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
            is_mtp_layer=True,
            draft_sliding_window=config.sliding_window,
            draft_rope_theta=(config.swa_rope_theta if config.sliding_window else None),
        )
        self.mlp = IQuestQ1MoEBlock(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            router_dtype=str_dtype_to_torch_dtype(config.moe_router_dtype),
            prefix=f"{prefix}.mlp",
        )
        self.attention_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.ffn_out_norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
        self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        dtype = hidden_states.dtype
        attention = self.self_attn(positions, self.attention_norm(hidden_states))
        residual = (
            hidden_states.float() if self.fp32_residual_connection else hidden_states
        )
        hidden_states = residual + self.attn_out_norm(attention) * self.attn_out_scale
        normalized = self.feed_forward_norm(hidden_states).to(dtype)
        output = self.mlp(normalized)
        return hidden_states + self.ffn_out_norm(output) * self.ffn_out_scale


class IQuestQ1RecursiveLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.enorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_model_layer = IQuestQ1RecursiveInnerLayer(
            vllm_config=vllm_config, prefix=f"{prefix}.mtp_model_layer"
        )
        self.final_layernorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions, hidden_states, inputs_embeds):
        dtype = self.eh_proj.weight.dtype
        hidden_states = self.eh_proj(
            torch.cat(
                [
                    self.enorm(inputs_embeds).to(dtype),
                    self.hnorm(hidden_states).to(dtype),
                ],
                dim=-1,
            )
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states)
        return self.final_layernorm(hidden_states).to(dtype)


class IQuestQ1RecursivePredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert vllm_config.speculative_config is not None
        target_config = vllm_config.speculative_config.target_model_config.hf_config
        self.layer_idx = target_config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleDict(
            {
                str(self.layer_idx): IQuestQ1RecursiveLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{self.layer_idx}",
                )
            }
        )

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[str(self.layer_idx)](positions, hidden_states, inputs_embeds)


@support_torch_compile
class IQuestQ1MTPRecursive(nn.Module, SupportsPP):
    """One draft layer recursively feeding its normalized hidden state back."""

    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        vllm_config = copy(vllm_config)
        assert vllm_config.speculative_config is not None
        validate_recursive_draft(vllm_config.speculative_config)
        vllm_config.model_config = vllm_config.speculative_config.draft_model_config
        vllm_config.quant_config = get_draft_quant_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        self.model = IQuestQ1RecursivePredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.mtp_start_layer_idx = self.model.layer_idx
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.head_dtype = (
            torch.float32
            if self.config.enable_lm_head_fp32
            else vllm_config.model_config.dtype
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None
        hidden_states = self.model(input_ids, positions, hidden_states, inputs_embeds)
        # EAGLE consumes separate logits and feedback states; ours are identical.
        return hidden_states, hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qkv_shards: set[str] = set()
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        enable_sink_attention = getattr(self.config, "enable_sink_attention", False)

        def _load_into(param_name: str, weight: torch.Tensor) -> None:
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            if name == "target_final_norm.weight":
                # The proposer already supplies post-norm target hidden states.
                continue
            if name == "embed_tokens.weight":
                _load_into("model.embed_tokens.weight", loaded_weight)
                continue
            if name == "lm_head.weight":
                _load_into(name, loaded_weight)
                continue
            if not name.startswith("mtp."):
                raise ValueError(f"Unexpected recursive MTP weight: {name}")
            for shard in ("q", "k", "v"):
                if name.endswith(f".self_attn.{shard}_proj.weight"):
                    qkv_shards.add(shard)
            name = f"model.layers.{self.mtp_start_layer_idx}." + name.removeprefix(
                "mtp."
            )

            # Fused QKV: match on the exact ".<component>.weight" suffix.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                suffix = f".{weight_name}.weight"
                if not name.endswith(suffix):
                    continue
                mapped = name[: -len(suffix)] + f".{param_name}.weight"
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                expert_weight_matched = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    expert_weight_matched = True
                    mapped = name.replace(weight_name, param_name)
                    if not is_pp_missing_parameter(mapped, self):
                        param = params_dict[mapped]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(mapped)
                    break

                if expert_weight_matched:
                    continue

                if name.endswith(".mlp.experts.fc"):
                    mapped = name[: -len(".fc")] + ".routed_experts.w13_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id][: self.config.intermediate_size],
                            mapped,
                            shard_id="w1",
                            expert_id=expert_id,
                        )
                        weight_loader(
                            param,
                            loaded_weight[expert_id][self.config.intermediate_size :],
                            mapped,
                            shard_id="w3",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue
                if name.endswith(".mlp.experts.proj"):
                    mapped = name[: -len(".proj")] + ".routed_experts.w2_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            mapped,
                            shard_id="w2",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue

                if not enable_sink_attention and name.endswith(".sink_k"):
                    logger.warning_once("sink attention feature is disabled")
                    continue

                if name.endswith(".mlp.router.weight"):
                    name = name[: -len(".router.weight")] + ".gate.weight"

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once("Unexpected MTP weight skipped: %s", name)
                    continue
                _load_into(name, loaded_weight)

        if qkv_shards != {"q", "k", "v"}:
            raise ValueError("Recursive MTP checkpoint is missing QKV projections")
        return loaded_params
