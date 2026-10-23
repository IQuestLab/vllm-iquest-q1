# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical check for the IQuestQ1 learned attention sink.

The sink contributes a key but no value, so it only grows the softmax
denominator. The paged kernel must match a dense reference that appends the
sink key explicitly.
"""

import pytest
import torch
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cuda():
    pytest.skip("IQuest sink attention requires CUDA", allow_module_level=True)

try:
    from vllm.vllm_flash_attn import is_fa_version_supported
except ImportError:
    pytest.skip("vllm_flash_attn is unavailable", allow_module_level=True)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 1024])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("query_scaling", ["none", "tensor", "head"])
@torch.inference_mode()
def test_sink_preserves_fp32_eager_rounding(
    num_tokens, output_dtype, strided, query_scaling
):
    """The sink correction must preserve rounding and leave padded rows untouched."""
    from vllm_iquest_q1.attention import apply_sink_key

    set_random_seed(42)
    num_heads, num_kv_heads, head_size = 6, 2, 128
    storage_heads = num_heads + int(strided)
    query = torch.randn(
        num_tokens + 3, storage_heads, head_size, device="cuda", dtype=torch.bfloat16
    )
    query_scale = None
    if query_scaling != "none":
        query = query.to(current_platform.fp8_dtype())
        if query_scaling == "head":
            query_scale = torch.tensor([0.25, -1, 0.5, -1], device="cuda")[::2]
            if not strided:
                query_scale = query_scale.contiguous()
        else:
            query_scale = torch.tensor(0.5, device="cuda")
    query = query[1:, :num_heads]
    sink = torch.randn(num_kv_heads, head_size, device="cuda", dtype=torch.bfloat16)
    output = torch.randn(
        num_tokens + 3, storage_heads, head_size, device="cuda", dtype=output_dtype
    )[1:, :num_heads]
    before = output.clone()
    repeated_sink = sink.repeat_interleave(num_heads // num_kv_heads, dim=0)
    scale = head_size**-0.5
    reference_query = query[:num_tokens].float()
    if query_scale is not None:
        scales = query_scale.expand(num_kv_heads).repeat_interleave(
            num_heads // num_kv_heads
        )
        reference_query = reference_query * scales[None, :, None]
    dot = (reference_query * repeated_sink.float()).sum(-1) * scale
    lse = (dot + torch.randn_like(dot)).T
    if not strided:
        lse = lse.contiguous()
    expected = (
        before[:num_tokens].float() * torch.sigmoid(lse.T - dot).unsqueeze(-1)
    ).to(output_dtype)

    apply_sink_key(query, sink, output, lse, scale, num_tokens, query_scale=query_scale)

    torch.testing.assert_close(output[:num_tokens], expected, rtol=0, atol=0)
    torch.testing.assert_close(output[num_tokens:], before[num_tokens:], rtol=0, atol=0)


@pytest.mark.parametrize("fa_version", [2, 3, 4])
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 2)])
@pytest.mark.parametrize("head_size", [72, 128])
@pytest.mark.parametrize("sliding_window", [None, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_lens", [[1, 3, 19], [1, 1, 1]])
@pytest.mark.parametrize(
    "kv_cache_dtype,per_head_scales",
    [
        ("auto", False),
        ("fp8", False),
        ("fp8_e4m3", False),
        ("fp8", True),
        ("fp8_e4m3", True),
    ],
)
@torch.inference_mode()
def test_iquest_paged_learned_sink_matches_dense_attention(
    fa_version: int,
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    query_lens: list[int],
    kv_cache_dtype: str,
    per_head_scales: bool,
) -> None:
    """The sink respects SWA, KV-head sharing, and non-unit FP8 Q/K/V scales."""
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is unavailable")
    quantized = kv_cache_dtype != "auto"
    if quantized and not (
        fa_version == 3
        and current_platform.is_device_capability_family(90)
        or fa_version == 4
        and current_platform.is_device_capability_family(100)
    ):
        pytest.skip("FP8 KV cache requires FA3 on SM90 or FA4 on SM100")
    if quantized and head_size % 16 != 0:
        pytest.skip("FP8 FlashAttention requires a head size divisible by 16")
    if quantized and dtype != torch.bfloat16:
        pytest.skip("FP8 FlashAttention outputs BF16")
    if (
        fa_version == 4
        and current_platform.is_device_capability(90)
        and head_size % 16 != 0
    ):
        pytest.skip("FA4 paged KV on SM90 requires a head size divisible by 16")

    from types import SimpleNamespace

    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    from vllm_iquest_q1.attention import (
        IQuestFlashAttentionBackend,
        IQuestFlashAttentionImpl,
    )

    set_random_seed(0)
    device = "cuda"
    kv_lens = [47, 31, 19]
    num_tokens = sum(query_lens)
    num_q_heads, num_kv_heads = num_heads
    block_size = 16
    scale = head_size**-0.5
    query = torch.randn(
        num_tokens + 5, num_q_heads + 1, head_size, dtype=dtype, device=device
    )[:, :num_q_heads]
    sink_key = 4 * torch.randn(num_kv_heads, head_size, dtype=dtype, device=device)
    kv_cache = torch.randn(
        9, num_kv_heads, block_size, 2 * head_size, dtype=dtype, device=device
    )
    block_table = torch.randperm(9, device=device).to(torch.int32).reshape(3, 3)
    query_start_loc = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        query_start_loc=query_start_loc,
        max_seq_len=max(kv_lens),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=torch.empty(num_tokens, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    impl = IQuestFlashAttentionImpl(
        num_q_heads,
        head_size,
        scale,
        num_kv_heads,
        None,
        sliding_window,
        kv_cache_dtype,
    )
    impl.vllm_flash_attn_version = fa_version
    impl.fa4_hd256 = False
    layer = SimpleNamespace(sink_key=sink_key)
    reference_query = query.float()
    reference_cache = kv_cache.float()
    if quantized:
        assert IQuestFlashAttentionBackend.supports_kv_cache_dtype(kv_cache_dtype)
        layer._q_scale = torch.tensor(0.5, device=device)
        layer._k_scale = torch.tensor(0.25, device=device)
        layer._v_scale = torch.tensor(2.0, device=device)
        if per_head_scales:
            layer._q_scale = torch.linspace(0.25, 0.5, num_kv_heads, device=device)
            layer._k_scale = torch.linspace(0.125, 0.25, num_kv_heads, device=device)
            layer._v_scale = torch.linspace(1.0, 2.0, num_kv_heads, device=device)
        q_scale = layer._q_scale.expand(num_kv_heads).repeat_interleave(
            num_q_heads // num_kv_heads
        )[None, :, None]
        k_scale = layer._k_scale.expand(num_kv_heads)[:, None, None]
        v_scale = layer._v_scale.expand(num_kv_heads)[:, None, None]
        fp8_dtype = current_platform.fp8_dtype()
        # Build the dense reference independently of the cache-write kernel.
        reference_query = (reference_query / q_scale).to(fp8_dtype).float() * q_scale
        reference_key, reference_value = reference_cache.chunk(2, dim=-1)
        reference_cache = torch.cat(
            [
                (reference_key / k_scale).to(fp8_dtype).float() * k_scale,
                (reference_value / v_scale).to(fp8_dtype).float() * v_scale,
            ],
            dim=-1,
        )
        query, _ = ops.scaled_fp8_quant(
            query.flatten(1).contiguous(),
            layer._q_scale,
            group_shape=(-1, num_q_heads * head_size // num_kv_heads)
            if per_head_scales
            else (-1, -1),
        )
        query = query.reshape(-1, num_q_heads, head_size)
        key, value = (
            kv_cache.transpose(1, 2)
            .reshape(-1, num_kv_heads, 2 * head_size)
            .chunk(2, dim=-1)
        )
        kv_cache = torch.empty_like(kv_cache, dtype=torch.uint8)
        impl.do_kv_cache_update(
            layer,
            # Cache writes require contiguous head and feature dimensions.
            key.contiguous(),
            value.contiguous(),
            kv_cache,
            torch.arange(9 * block_size, device=device, dtype=torch.int64),
        )
        cached_key, cached_value = kv_cache.view(fp8_dtype).float().chunk(2, dim=-1)
        torch.testing.assert_close(
            torch.cat([cached_key * k_scale, cached_value * v_scale], dim=-1),
            reference_cache,
            atol=0,
            rtol=0,
        )
    output = torch.full(
        (num_tokens + 5, num_q_heads + 1, head_size),
        -73,
        dtype=dtype,
        device=device,
    )[:, :num_q_heads]
    impl.forward(
        layer,
        query,
        None,
        None,
        kv_cache,
        metadata,
        output,
    )

    repeat = num_q_heads // num_kv_heads
    sink = sink_key.float().repeat_interleave(repeat, dim=0)
    start = 0
    for seq, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        cache = (
            reference_cache[block_table[seq]]
            .transpose(1, 2)
            .reshape(-1, num_kv_heads, 2 * head_size)[:kv_len]
        )
        key, value = cache.float().repeat_interleave(repeat, dim=1).chunk(2, dim=-1)
        q = reference_query[start : start + query_len]
        logits = torch.einsum("qhd,khd->hqk", q, key) * scale
        query_pos = torch.arange(kv_len - query_len, kv_len, device=device)
        key_pos = torch.arange(kv_len, device=device)
        visible = key_pos[None, :] <= query_pos[:, None]
        if sliding_window is not None:
            visible &= key_pos[None, :] > query_pos[:, None] - sliding_window
        logits.masked_fill_(~visible[None, :, :], -torch.inf)
        sink_logits = torch.einsum("qhd,hd->hq", q, sink) * scale
        probabilities = torch.cat([logits, sink_logits[..., None]], dim=-1).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probabilities[..., :-1], value)
        if quantized:
            # Match vLLM's test_attention_backends.py FP8 SDPA tolerances;
            # FlashAttention also rounds the attention probabilities to FP8.
            atol, rtol = 6e-2, 1e-1
        else:
            atol = rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(
            output[start : start + query_len].float(),
            expected,
            atol=atol,
            rtol=rtol,
        )
        start += query_len
    assert torch.all(output[num_tokens:] == -73)


@pytest.mark.parametrize("cudagraph", [False, True])
@torch.inference_mode()
def test_fp8_attention_layer_quantizes_query_and_applies_sink(cudagraph):
    """Exercise query quantization and cache writes through the real layer API."""
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.fa_utils import flash_attn_supports_kv_cache_dtype
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    from vllm_iquest_q1.attention import IQuestAttention

    if not flash_attn_supports_kv_cache_dtype("fp8", head_size=128):
        pytest.skip("FP8 KV cache is unavailable")

    set_random_seed(13)
    num_tokens, num_heads, num_kv_heads, head_size = 2, 6, 2, 128
    scale = head_size**-0.5
    config = VllmConfig(
        cache_config=CacheConfig(cache_dtype="fp8", block_size=16),
        compilation_config=CompilationConfig(mode=0),
    )
    prefix = "model.layers.0.self_attn.attn"
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        sink = torch.randn(num_kv_heads, head_size)
        layer = IQuestAttention(
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
            cache_config=config.cache_config,
            prefix=prefix,
            sink_key=sink,
        )
        layer._q_scale.fill_(0.5)
        layer._k_scale.fill_(0.25)
        layer._v_scale.fill_(2.0)
        layer.kv_cache = torch.empty(
            2, num_kv_heads, 16, 2 * head_size, dtype=torch.uint8
        )
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_kv_heads * head_size)
        value = torch.randn_like(key)
        metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            max_seq_len=1,
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            block_table=torch.tensor([[1], [0]], dtype=torch.int32),
            slot_mapping=torch.tensor([16, 0], dtype=torch.int64),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        with set_forward_context(
            {prefix: metadata},
            config,
            slot_mapping={prefix: metadata.slot_mapping},
        ):
            output = layer(query, key, value)
            if cudagraph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = layer(query, key, value)
                # Replay must read the new queries and scales from device memory.
                query.mul_(0.5)
                layer._q_scale.fill_(0.25)
                graph.replay()

        fp8_dtype = current_platform.fp8_dtype()
        q = (query.float() / layer._q_scale).to(fp8_dtype).float() * layer._q_scale
        k = (key.float() / layer._k_scale).to(fp8_dtype).float() * layer._k_scale
        v = (value.float() / layer._v_scale).to(fp8_dtype).float() * layer._v_scale
        q = q.reshape(num_tokens, num_heads, head_size)
        repeat = num_heads // num_kv_heads
        k = k.reshape(num_tokens, num_kv_heads, head_size).repeat_interleave(repeat, 1)
        v = v.reshape(num_tokens, num_kv_heads, head_size).repeat_interleave(repeat, 1)
        sink = sink.float().repeat_interleave(repeat, 0)
        # Each sequence has one real token plus the zero-valued learned sink.
        probability = torch.sigmoid(((q * k).sum(-1) - (q * sink).sum(-1)) * scale)
        expected = (v * probability.unsqueeze(-1)).flatten(1)
        torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)
