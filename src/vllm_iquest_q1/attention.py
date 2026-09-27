# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged attention with an iQuest learned key and zero-valued sink."""

from typing import Any, ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention import Attention
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flash_attn as fa
from vllm.v1.attention.backends.fa_utils import FA4_HD256_PAGE_SIZE


@triton.jit(do_not_specialize=["num_tokens"])
def _apply_sink_key_kernel(
    query,
    sink_key,
    output,
    lse,
    num_tokens,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    sink_stride_h: tl.constexpr,
    sink_stride_d: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_d: tl.constexpr,
    lse_stride_h: tl.constexpr,
    lse_stride_t: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tokens = rows // NUM_HEADS
    heads = rows % NUM_HEADS
    dims = tl.arange(0, BLOCK_D)
    valid = (tokens[:, None] < num_tokens) & (dims[None, :] < HEAD_SIZE)
    q = tl.load(
        query
        + tokens[:, None] * query_stride_t
        + heads[:, None] * query_stride_h
        + dims[None, :] * query_stride_d,
        valid,
        0,
    ).to(tl.float32)
    sink = tl.load(
        sink_key
        + (heads[:, None] // QUERIES_PER_KV) * sink_stride_h
        + dims[None, :] * sink_stride_d,
        valid,
        0,
    ).to(tl.float32)
    products = q * sink
    if HEAD_SIZE == 128:
        # Match the FP32 eager reduction order for the model's head size.
        even, odd = tl.split(products.reshape(BLOCK_ROWS, HEAD_SIZE // 2, 2))
        first, third = tl.split(even.reshape(BLOCK_ROWS, HEAD_SIZE // 4, 2))
        second, fourth = tl.split(odd.reshape(BLOCK_ROWS, HEAD_SIZE // 4, 2))
        sink_dot = tl.sum(((first + second) + third) + fourth, axis=1)
    else:
        sink_dot = tl.sum(products, axis=1)
    sink_logit = sink_dot * SCALE
    normal_lse = tl.load(
        lse + heads * lse_stride_h + tokens * lse_stride_t,
        tokens < num_tokens,
        0,
    )
    # Adding a zero-valued sink changes only the softmax denominator.
    factor = tl.div_rn(1.0, 1.0 + tldevice.exp(sink_logit - normal_lse))
    offsets = (
        tokens[:, None] * output_stride_t
        + heads[:, None] * output_stride_h
        + dims[None, :] * output_stride_d
    )
    values = tl.load(output + offsets, valid, 0).to(tl.float32)
    tl.store(output + offsets, values * factor[:, None], valid)


def apply_sink_key(
    query: torch.Tensor,
    sink_key: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    num_tokens: int,
) -> None:
    """Include an always-visible zero-valued sink in paged attention output."""
    if num_tokens == 0:
        return
    num_heads, head_size = query.shape[1:]
    _apply_sink_key_kernel[(cdiv(num_tokens * num_heads, 4),)](
        query,
        sink_key,
        output,
        lse,
        num_tokens,
        *query.stride(),
        *sink_key.stride(),
        *output.stride(),
        *lse.stride(),
        NUM_HEADS=num_heads,
        QUERIES_PER_KV=num_heads // sink_key.shape[0],
        HEAD_SIZE=head_size,
        SCALE=scale,
        BLOCK_ROWS=4,
        BLOCK_D=triton.next_power_of_2(head_size),
        num_warps=4,
        enable_fp_fusion=False,
    )


class IQuestFlashAttentionMetadataBuilder(fa.FlashAttentionMetadataBuilder):
    def use_cascade_attention(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def build(
        self, common_prefix_len: int, *args: Any, **kwargs: Any
    ) -> fa.FlashAttentionMetadata:
        return super().build(0, *args, **kwargs)


class IQuestFlashAttentionBackend(fa.FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_impl_cls() -> type["IQuestFlashAttentionImpl"]:
        return IQuestFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[IQuestFlashAttentionMetadataBuilder]:
        return IQuestFlashAttentionMetadataBuilder


class IQuestFlashAttentionImpl(fa.FlashAttentionImpl):
    supports_dcp = False
    can_return_lse_for_decode = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "iQuest learned sinks require unquantized KV cache"
            )
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError("iQuest learned sinks require decoder attention")
        if self.alibi_slopes is not None or self.logits_soft_cap:
            raise NotImplementedError(
                "iQuest learned sinks do not support ALiBi or attention soft caps"
            )
        if self.sinks is not None:
            raise ValueError("iQuest learned keys cannot use per-head constant sinks")
        config = get_current_vllm_config_or_none()
        if config is not None:
            parallel = config.parallel_config
            if (
                parallel.decode_context_parallel_size > 1
                or parallel.prefill_context_parallel_size > 1
            ):
                raise NotImplementedError(
                    "iQuest learned sinks do not support context parallelism"
                )
        self.supports_quant_query_input = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: fa.FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "iQuest learned sinks do not support output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if (
            attn_metadata.use_cascade
            or attn_metadata.mm_prefix_query_range_tensor is not None
            or attn_metadata.rswa_prefix_lens is not None
            or attn_metadata.causal is not True
        ):
            raise NotImplementedError(
                "iQuest learned sinks require causal full or sliding-window attention"
            )
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        max_seq_len = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        num_splits = attn_metadata.max_num_splits
        if self.fa4_hd256:
            num_pages = cdiv(max_seq_len, FA4_HD256_PAGE_SIZE)
            max_seq_len = num_pages * FA4_HD256_PAGE_SIZE
            block_table = block_table[:, :num_pages]
            num_splits = 1
        assert self.vllm_flash_attn_version is not None
        _, lse = fa.flash_attn_varlen_func(
            q=query[:num_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=list(self.sliding_window),
            block_table=block_table,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            num_splits=num_splits,
            return_softmax_lse=True,
        )
        apply_sink_key(query, layer.sink_key, output, lse, self.scale, num_tokens)
        return output


class IQuestAttention(Attention):
    """Attention with a learned sink key owned by the parent model layer."""

    def __init__(self, *args: Any, sink_key: torch.Tensor, **kwargs: Any) -> None:
        kwargs["attn_backend"] = IQuestFlashAttentionBackend
        super().__init__(*args, **kwargs)
        if self.head_size_v != self.head_size:
            raise NotImplementedError(
                "iQuest learned sinks require equal query, key, and value head sizes"
            )
        if sink_key.shape != (self.num_kv_heads, self.head_size):
            raise ValueError(
                "iQuest sink key must have shape "
                f"({self.num_kv_heads}, {self.head_size}), got {sink_key.shape}"
            )
        # The parent owns this parameter and its tensor-parallel weight loader.
        object.__setattr__(self, "sink_key", sink_key)
