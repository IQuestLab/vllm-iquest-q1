# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the IQuestQ1 draft head's checkpoint and residual contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm_iquest_q1 import mtp_recursive
from vllm_iquest_q1.configs import IQuestQ1Config, IQuestQ1MTPConfig

from vllm.transformers_utils.configs.eagle import EAGLEConfig


def _speculative_config():
    target = IQuestQ1Config()
    draft = IQuestQ1MTPConfig(
        target_config=target.to_dict(), architectures=["IQuestQ1MTP"]
    )
    return SimpleNamespace(
        method="eagle",
        parallel_drafting=False,
        num_speculative_tokens=7,
        target_model_config=SimpleNamespace(hf_config=target),
        draft_model_config=SimpleNamespace(
            hf_config=EAGLEConfig(draft, method="eagle", model_type="eagle")
        ),
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("method", "eagle3", "requires method='eagle'"),
        ("parallel_drafting", True, "serial drafting"),
        ("hidden_size", 4096, "must match"),
        ("vocab_size", 32000, "must match"),
        ("model_type", "llama", "IQuestQ1 target"),
    ],
)
def test_recursive_plugin_rejects_incompatible_target_or_proposer(field, value, error):
    config = _speculative_config()
    mtp_recursive.validate_recursive_draft(config)
    obj = config if hasattr(config, field) else config.target_model_config.hf_config
    setattr(obj, field, value)
    with pytest.raises(ValueError, match=error):
        mtp_recursive.validate_recursive_draft(config)


def test_recursive_eagle_returns_same_normalized_state_for_logits_and_feedback():
    """The EAGLE tuple must preserve the previous single-tensor feedback rule."""
    expected = torch.randn(3, 4)
    draft = SimpleNamespace(model=lambda *args: expected)
    logits_state, feedback = mtp_recursive.IQuestQ1MTPRecursive.forward(
        draft, torch.tensor([1, 2, 3]), torch.arange(3), torch.randn(3, 4)
    )
    assert logits_state is expected
    assert feedback is expected


class _Attention(nn.Module):
    def forward(self, positions, hidden_states):
        return (hidden_states.roll(1, dims=-1) + positions[:, None] * 0.1).to(
            hidden_states.dtype
        )


@pytest.fixture
def tiny_config(monkeypatch):
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=3,
        num_experts=2,
        num_experts_per_tok=1,
        moe_router_dtype="float32",
        rms_norm_eps=1e-5,
        first_layer_attn_out_scale=0.7,
        first_layer_ffn_out_scale=0.4,
        attn_out_scale=1.3,
        ffn_out_scale=1.7,
        num_hidden_layers=42,
        vocab_size=8,
        sliding_window=512,
        swa_rope_theta=10000.0,
        fp32_residual_connection=True,
    )
    monkeypatch.setattr(
        mtp_recursive, "IQuestQ1Attention", lambda **kwargs: _Attention()
    )
    monkeypatch.setattr(mtp_recursive, "IQuestQ1MoEBlock", lambda **kwargs: nn.SiLU())
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        speculative_config=SimpleNamespace(
            target_model_config=SimpleNamespace(hf_config=config)
        ),
    )


def _rms_norm(x, module):
    return F.rms_norm(
        x.float(), (x.shape[-1],), module.weight.float(), module.variance_epsilon
    ).to(x.dtype)


def test_recursive_steps_reuse_one_draft_layer(tiny_config, monkeypatch):
    """Recursive steps reuse the draft layer and feed back normalized states."""
    monkeypatch.setattr(
        mtp_recursive,
        "VocabParallelEmbedding",
        lambda vocab_size, hidden_size, **kwargs: nn.Embedding(vocab_size, hidden_size),
    )
    torch.manual_seed(0)
    model = mtp_recursive.IQuestQ1RecursivePredictor(vllm_config=tiny_config)
    ids = torch.tensor([1, 3, 2])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, 4)
    embeds = model.embed_tokens(ids)
    first = model(ids, positions, hidden)
    second = model(None, positions, first, embeds)

    assert list(model.layers) == ["42"]
    torch.testing.assert_close(first, model.layers["42"](positions, hidden, embeds))
    torch.testing.assert_close(second, model.layers["42"](positions, first, embeds))
    assert not torch.allclose(second, first)


@pytest.mark.parametrize("fused_experts", [False, True])
def test_recursive_loads_offset_layers_and_fused_weights(fused_experts):
    """Standalone weights load with correct layer offsets and expert/QKV shards."""
    params = {}
    checkpoint = []
    expected = {}

    def add_served_parameter(name, value, loader=None):
        param = nn.Parameter(torch.zeros_like(value), requires_grad=False)
        if loader is not None:
            param.weight_loader = loader
        params[name] = param
        expected[name] = value

    def qkv_loader(param, value, shard_id):
        param[{"q": 0, "k": 1, "v": 2}[shard_id]].copy_(value)

    def expert_loader(param, value, name, shard_id, expert_id):
        if shard_id == "w2":
            param[expert_id].copy_(value)
        else:
            param[expert_id, {"w1": 0, "w3": 1}[shard_id]].copy_(value)

    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        value = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        add_served_parameter(name, value)
        source_name = name.removeprefix("model.")
        checkpoint.append((source_name, value))
    src = "mtp"
    dst = "model.layers.42"
    qkv = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    add_served_parameter(
        f"{dst}.mtp_model_layer.self_attn.qkv_proj.weight", qkv, qkv_loader
    )
    for shard, value in zip(("q", "k", "v"), qkv):
        checkpoint.append(
            (f"{src}.mtp_model_layer.self_attn.{shard}_proj.weight", value)
        )

    w13 = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    w2 = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    for name, value in (("w13_weight", w13), ("w2_weight", w2)):
        add_served_parameter(
            f"{dst}.mtp_model_layer.mlp.experts.routed_experts.{name}",
            value,
            expert_loader,
        )
    if fused_experts:
        checkpoint.extend(
            [
                (f"{src}.mtp_model_layer.mlp.experts.fc", w13.reshape(2, 6, 4)),
                (f"{src}.mtp_model_layer.mlp.experts.proj", w2),
            ]
        )
    else:
        for expert_id in range(2):
            for projection, value in (
                ("gate_proj", w13[expert_id, 0]),
                ("up_proj", w13[expert_id, 1]),
                ("down_proj", w2[expert_id]),
            ):
                checkpoint.append(
                    (
                        f"{src}.mtp_model_layer.mlp.experts.{expert_id}."
                        f"{projection}.weight",
                        value,
                    )
                )
    router = torch.full((2, 4), 0.25)
    add_served_parameter(f"{dst}.mtp_model_layer.mlp.gate.weight", router)
    checkpoint.append((f"{src}.mtp_model_layer.mlp.router.weight", router))
    norm = torch.full((4,), 0.5)
    add_served_parameter(f"{dst}.final_layernorm.weight", norm)
    checkpoint.append((f"{src}.final_layernorm.weight", norm))

    checkpoint.append(("target_final_norm.weight", torch.ones(4)))
    draft = nn.Module()
    draft.config = SimpleNamespace(num_experts=2, intermediate_size=3)
    draft.mtp_start_layer_idx = 42
    draft.named_parameters = lambda: iter(params.items())
    model_cls = mtp_recursive.IQuestQ1MTPRecursive
    loaded = model_cls.load_weights(draft, checkpoint)

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(params[name], value, msg=name)

    loaded = model_cls.load_weights(draft, checkpoint[1:])
    assert loaded == set(expected) - {"model.embed_tokens.weight"}
    with pytest.raises(ValueError, match="missing QKV"):
        model_cls.load_weights(
            draft,
            [(name, weight) for name, weight in checkpoint if ".q_proj." not in name],
        )


@pytest.mark.parametrize("fp32_residual", [False, True])
def test_recursive_mtp_keeps_bos_embedding_and_residual_precision(
    tiny_config, fp32_residual
):
    """A recursive step consumes real shifted tokens, including at position zero."""
    tiny_config.model_config.hf_config.fp32_residual_connection = fp32_residual
    torch.manual_seed(7)
    layer = mtp_recursive.IQuestQ1RecursiveLayer(vllm_config=tiny_config).bfloat16()
    positions = torch.tensor([0, 511, 512])
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    embeds = torch.randn(3, 4, dtype=torch.bfloat16)
    projected = F.linear(
        torch.cat([_rms_norm(embeds, layer.enorm), _rms_norm(hidden, layer.hnorm)], -1),
        layer.eh_proj.weight,
    )
    inner = layer.mtp_model_layer
    attention = inner.self_attn(positions, _rms_norm(projected, inner.attention_norm))
    residual = projected.float() if fp32_residual else projected
    residual = residual + _rms_norm(attention, inner.attn_out_norm) * 0.7
    ffn = F.silu(_rms_norm(residual, inner.feed_forward_norm).bfloat16())
    expected = _rms_norm(
        residual + _rms_norm(ffn, inner.ffn_out_norm) * 0.4,
        layer.final_layernorm,
    ).bfloat16()
    actual = layer(positions, hidden, embeds)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    zeroed = embeds.clone()
    zeroed[0] = 0
    assert not torch.equal(actual[0], layer(positions, hidden, zeroed)[0])
