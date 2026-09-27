# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IQuestQ1 hybrid configuration, routing precision, and checkpoint layout tests."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm_iquest_q1 import model as iquest_model
from vllm_iquest_q1.configs import IQuestQ1Config, IQuestQ1MTPConfig
from vllm_iquest_q1.model import (
    IQuestQ1Attention,
    IQuestQ1ForCausalLM,
    IQuestQ1Model,
    IQuestQ1MoEBlock,
    IQuestQ1RMSNorm,
    get_layer_sliding_window_size,
)

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.config import get_config


@pytest.mark.cpu_test
def test_installed_entry_point_registers_models_and_parsers_in_fresh_process():
    """Workers must discover the plugin without in-tree registry hooks."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import SimpleNamespace
import torch
from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.plugins import load_general_plugins
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
load_general_plugins()
from vllm_iquest_q1 import register
register()
assert 'vllm_iquest_q1.model' not in sys.modules
assert 'vllm_iquest_q1.mtp_recursive' not in sys.modules
assert not torch.cuda.is_initialized()
assert AutoConfig.for_model('iquest_q1').num_hidden_layers == 88
assert AutoConfig.for_model('iquest_q1_mtp').num_hidden_layers == 1
for arch in ('IQuestQ1ForCausalLM', 'IQuestQ1MTP', 'EagleIQuestQ1MTP'):
    cls, _ = ModelRegistry.resolve_model_cls([arch], SimpleNamespace(model_impl="vllm"))
    assert cls.__module__.startswith('vllm_iquest_q1.')
assert ReasoningParserManager.get_reasoning_parser('iquest_q1')
assert ToolParserManager.get_tool_parser('iquest_q1')
""",
        ],
        env={**os.environ, "VLLM_PLUGINS": "iquest_q1"},
        check=True,
        timeout=60,
    )


@pytest.mark.cpu_test
def test_recursive_draft_preserves_own_attention_config(tmp_path):
    target = IQuestQ1Config()
    config = dict(
        model_type="iquest_q1_mtp",
        target_config=target.to_dict(),
        architectures=["IQuestQ1MTP"],
        sliding_window=512,
        swa_rope_theta=10000.0,
        num_draft_slots=7,
        fp32_residual_connection=True,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(loaded, IQuestQ1MTPConfig)
    assert loaded.model_type == "iquest_q1_mtp"
    assert loaded.num_hidden_layers == 1
    assert loaded.layer_types == ["sliding_attention"]
    assert loaded.sliding_window == 512
    assert loaded.swa_rope_theta == 10000.0
    assert loaded.fp32_residual_connection
    assert loaded.target_config["num_hidden_layers"] == 88
    assert loaded.target_config["sliding_window"] == 4096
    assert loaded.num_draft_slots == 7
    assert SpeculativeConfig.hf_config_override(loaded).architectures == ["IQuestQ1MTP"]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs",
    [
        {"draft_type": "diffusion"},
        {"dense_ffn": True},
        {"eagle3_shared_kv": True},
        {"dflash_num_layers": 2},
        {"sliding_window": 512},
        {"num_draft_slots": 0},
    ],
)
def test_recursive_config_rejects_unsupported_drafts(kwargs):
    with pytest.raises(ValueError):
        IQuestQ1MTPConfig(**kwargs)


@pytest.mark.cpu_test
@pytest.mark.parametrize("window", [None, 512])
def test_recursive_attention_uses_draft_window_and_rope(monkeypatch, window):
    """A draft layer after the target must retain its own window and RoPE base."""
    observed = {}

    def attention(*args, **kwargs):
        observed["window"] = kwargs["per_layer_sliding_window"]
        return nn.Identity()

    def rope(*args, **kwargs):
        observed["theta"] = kwargs["rope_parameters"]["rope_theta"]
        return nn.Identity()

    monkeypatch.setattr(iquest_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(iquest_model, "get_tensor_model_parallel_rank", lambda: 0)
    for name in ("QKVParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(iquest_model, name, lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(iquest_model, "Attention", attention)
    monkeypatch.setattr(iquest_model, "get_rope", rope)
    config = IQuestQ1MTPConfig(
        target_config=IQuestQ1Config(enable_sink_attention=False).to_dict(),
        sliding_window=window,
        swa_rope_theta=10000.0,
    )
    IQuestQ1Attention(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            cache_config=None,
            quant_config=None,
        ),
        prefix="draft.layers.88.self_attn",
        is_mtp_layer=True,
        draft_sliding_window=window,
        draft_rope_theta=10000.0 if window else None,
    )
    assert observed == {"window": window, "theta": 10000.0 if window else 1000000.0}


@pytest.mark.cpu_test
def test_hybrid_config_roundtrip_preserves_global_and_windowed_layers(tmp_path):
    config = IQuestQ1Config(architectures=["IQuestQ1ForCausalLM"])
    config.save_pretrained(tmp_path)
    loaded = get_config(str(tmp_path), trust_remote_code=False)
    assert loaded.max_position_embeddings == 524288
    assert loaded.layer_types == (
        ["full_attention"]
        + [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        * 21
        + ["full_attention"] * 3
    )
    assert loaded.layer_types.count("full_attention") == 25
    assert loaded.layer_types.count("sliding_attention") == 63
    assert loaded.rope_parameters["partial_rotary_factor"] == 0.25
    assert loaded.rope_parameters["rope_theta"] == 1000000.0
    assert loaded.swa_rope_theta == 10000.0


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "layer_idx,expected",
    [(0, None), (1, None), (2, 4096), (84, 4096), (85, None), (87, None)],
)
def test_layer_windows_match_checkpoint_pattern(layer_idx, expected):
    config = IQuestQ1Config()
    assert (
        get_layer_sliding_window_size(
            config.first_layers_types,
            config.last_layers_types,
            config.hybrid_layers_types_block,
            config.num_hybrid_layers_block,
            layer_idx,
            config.sliding_window,
        )
        == expected
    )
    assert (
        get_layer_sliding_window_size(
            config.first_layers_types,
            config.last_layers_types,
            config.hybrid_layers_types_block,
            config.num_hybrid_layers_block,
            88,
            config.sliding_window,
            is_mtp_layer=True,
        )
        is None
    )


@pytest.mark.cpu_test
def test_invalid_hybrid_pattern_fails_before_weight_loading():
    with pytest.raises(ValueError, match="hybrid layer pattern"):
        IQuestQ1Config(num_hidden_layers=87)


@pytest.mark.cpu_test
def test_rms_norm_multiplies_weight_before_casting_to_activation_dtype():
    generator = torch.Generator().manual_seed(42)
    norm = IQuestQ1RMSNorm(128).to(dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, generator=generator))
    x = torch.randn(7, 128, generator=generator).to(torch.bfloat16)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    expected = (normalized * norm.weight.float()).to(x.dtype)
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name,router_dtype",
    [
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("float16", torch.float16),
    ],
)
def test_router_dtype_preserves_expert_activation_dtype(
    monkeypatch, name, router_dtype
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.fused_moe.router import gate_linear
    from vllm.utils.torch_utils import set_default_torch_dtype

    captured = {}

    class Experts(nn.Module):
        def forward(self, hidden_states, router_logits):
            captured["expert_dtype"] = hidden_states.dtype
            captured["logits"] = router_logits
            return hidden_states

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        gate_linear,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False,
            is_rocm=lambda: False,
            is_device_capability=lambda _: False,
            is_device_capability_family=lambda _: False,
        ),
    )
    monkeypatch.setattr(iquest_model, "FusedMoEFactory", lambda **kwargs: Experts())
    config = IQuestQ1Config(moe_router_dtype=name)
    with set_default_torch_dtype(torch.bfloat16):
        block = IQuestQ1MoEBlock(
            8, 2, 16, 32, router_dtype=getattr(torch, config.moe_router_dtype)
        )
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(8, 16, generator=generator).bfloat16()
    block.gate.weight_loader(block.gate.weight, weight)
    x = torch.randn(3, 16, generator=generator).bfloat16()
    torch.testing.assert_close(block(x), x)
    assert block.gate.weight.dtype == router_dtype
    assert captured["expert_dtype"] == torch.bfloat16
    assert captured["logits"].dtype == router_dtype
    torch.testing.assert_close(
        captured["logits"],
        torch.nn.functional.linear(x.to(router_dtype), weight.to(router_dtype)),
        rtol=0,
        atol=0,
    )


@pytest.mark.cpu_test
def test_invalid_router_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unsupported moe_router_dtype"):
        IQuestQ1Config(moe_router_dtype="fp64")


@pytest.mark.cpu_test
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("enable_fp32", [False, True])
@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
def test_lm_head_precision_uses_own_flag(monkeypatch, draft, enable_fp32, model_dtype):
    """The draft must not inherit the target head's FP32 flag or global dtype."""
    from vllm_iquest_q1 import mtp_recursive

    from vllm.model_executor.layers import logits_processor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    target = IQuestQ1Config(
        hidden_size=16, vocab_size=8, enable_lm_head_fp32=True if draft else enable_fp32
    )
    config = target
    if draft:
        config = IQuestQ1MTPConfig(
            target_config=target.to_dict(),
            **({"enable_lm_head_fp32": True} if enable_fp32 else {}),
        )
    assert config.enable_lm_head_fp32 == enable_fp32
    model_config = SimpleNamespace(hf_config=config, dtype=model_dtype)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        speculative_config=SimpleNamespace(
            draft_model_config=model_config,
            target_model_config=SimpleNamespace(hf_config=target),
            method="eagle",
            parallel_drafting=False,
            num_speculative_tokens=7,
        ),
        compilation_config=SimpleNamespace(mode=0),
    )
    backbone = nn.Identity()
    backbone.layer_idx = 88
    backbone.make_empty_intermediate_tensors = lambda *a, **kw: {}
    head = nn.Linear(16, 8, bias=False, dtype=model_dtype)
    head.tp_size = 1
    head.quant_method = UnquantizedEmbeddingMethod()
    monkeypatch.setattr(
        iquest_model, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    monkeypatch.setattr(iquest_model, "IQuestQ1Model", lambda **kw: backbone)
    monkeypatch.setattr(
        mtp_recursive, "IQuestQ1RecursivePredictor", lambda **kw: backbone
    )
    monkeypatch.setattr(mtp_recursive, "get_draft_quant_config", lambda _: None)
    for module in (iquest_model, mtp_recursive):
        monkeypatch.setattr(module, "ParallelLMHead", lambda *a, **kw: head)
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(head_dtype=torch.float32)),
    )
    cls = mtp_recursive.IQuestQ1MTPRecursive if draft else IQuestQ1ForCausalLM
    model = cls(vllm_config=vllm_config)
    hidden = torch.randn(3, 16, dtype=model_dtype)
    logits = model.compute_logits(hidden)
    expected_dtype = torch.float32 if enable_fp32 else model_dtype
    assert logits.dtype == expected_dtype
    assert model.lm_head.weight.dtype == model_dtype
    torch.testing.assert_close(
        logits,
        torch.nn.functional.linear(
            hidden.to(expected_dtype), head.weight.to(expected_dtype)
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32, 128, 1024])
@torch.inference_mode()
def test_gate_linear_preserves_fp32_logits_with_model_dtype_weights(
    monkeypatch, num_tokens
):
    """Keep model-dtype router weights on supported GPUs without BF16 logits."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.utils.torch_utils import set_default_torch_dtype

    captured = {}

    class Experts(nn.Module):
        def forward(self, hidden_states, router_logits):
            captured["logits"] = router_logits
            return hidden_states

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(iquest_model, "FusedMoEFactory", lambda **kwargs: Experts())
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with set_default_torch_dtype(torch.bfloat16):
        block = IQuestQ1MoEBlock(256, 8, 3072, 1536).cuda()
    expected_weight_dtype = (
        torch.bfloat16 if block.gate.allow_specialized_router_gemm else torch.float32
    )
    assert block.gate.weight.dtype == expected_weight_dtype
    generator = torch.Generator(device="cuda").manual_seed(42)
    weight = torch.randn(256, 3072, generator=generator, device="cuda").bfloat16()
    block.gate.weight_loader(block.gate.weight, weight)
    x = torch.randn(num_tokens, 3072, generator=generator, device="cuda").bfloat16()
    expected = torch.nn.functional.linear(x.float(), weight.float())

    torch.testing.assert_close(block(x), x, rtol=0, atol=0)
    assert captured["logits"].dtype == torch.float32
    torch.testing.assert_close(captured["logits"], expected, rtol=1e-4, atol=1e-3)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp_size,kv_heads", [(8, 8), (8, 2), (2, 8)])
def test_sink_weights_follow_kv_head_sharding_and_replication(tp_size, kv_heads):
    weight = torch.arange(kv_heads * 4).reshape(kv_heads, 4).float()
    local_heads = max(1, kv_heads // tp_size)
    for rank in range(tp_size):
        attention = SimpleNamespace(
            tp_size=tp_size,
            tp_rank=rank,
            total_num_kv_heads=kv_heads,
            num_kv_heads=local_heads,
        )
        param = nn.Parameter(torch.empty(local_heads, 4), requires_grad=False)
        IQuestQ1Attention.sinks_k_weight_loader(attention, param, weight)
        first_head = rank * kv_heads // tp_size
        torch.testing.assert_close(param, weight[first_head : first_head + local_heads])


@pytest.mark.cpu_test
def test_base_weight_loader_loads_backbone_and_head_but_skips_mtp():
    model = IQuestQ1ForCausalLM.__new__(IQuestQ1ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Linear(2, 2, bias=False)
    model.lm_head = nn.Linear(2, 3, bias=False)
    backbone = torch.arange(4).reshape(2, 2).float()
    head = torch.arange(6).reshape(3, 2).float()

    loaded = model.load_weights(
        [
            ("model.weight", backbone),
            ("mtp_layers.0.eh_proj.weight", torch.zeros(2, 4)),
            ("lm_head.weight", head),
        ]
    )

    assert loaded == {"model.weight", "lm_head.weight"}
    torch.testing.assert_close(model.model.weight, backbone)
    torch.testing.assert_close(model.lm_head.weight, head)


@pytest.mark.cpu_test
def test_expert_checkpoint_splits_gate_up_and_maps_down_weights():
    model = IQuestQ1Model.__new__(IQuestQ1Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=2, intermediate_size=3)
    model.use_oe_embedding = False
    model.enable_sink_attention = True
    model.get_expert_mapping = lambda: []
    loaded = {}

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.empty(2, 6, 4), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(2, 4, 3), requires_grad=False)

    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.routed_experts = Experts()
    model.layers = nn.ModuleList([layer])

    def load(param, tensor, name, shard_id, expert_id):
        loaded[(name, shard_id, expert_id)] = tensor.clone()

    for param in model.parameters():
        param.weight_loader = load
    fc = torch.arange(48).reshape(2, 6, 4).float()
    proj = torch.arange(24).reshape(2, 4, 3).float()
    names = model.load_weights(
        [
            ("layers.0.mlp.experts.fc", fc),
            ("layers.0.mlp.experts.proj", proj),
        ]
    )
    w13 = "layers.0.mlp.experts.routed_experts.w13_weight"
    w2 = "layers.0.mlp.experts.routed_experts.w2_weight"
    assert names == {w13, w2}
    for expert in range(2):
        torch.testing.assert_close(loaded[w13, "w1", expert], fc[expert, :3])
        torch.testing.assert_close(loaded[w13, "w3", expert], fc[expert, 3:])
        torch.testing.assert_close(loaded[w2, "w2", expert], proj[expert])
