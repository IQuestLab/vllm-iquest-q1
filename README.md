# vllm-iquest-q1

A vLLM plugin for IQuest-Q1, with recursive MTP, reasoning parsing, and tool calling.

## Installation

Tested with vLLM commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`. 
With a compatible vLLM environment activated, install from this repository:

```bash
uv pip install --no-deps .
```

Or use docker:
```
docker pull iquestlabworkspace/vllm-iquest-q1:cu130
```

vLLM loads the plugin automatically. Install it in every worker environment.

## Serving

```bash
MODEL_ROOT="$(hf download IQuestLab/IQuest-Q1 --quiet)" && \
vllm serve "$MODEL_ROOT" \
  --served-model-name IQuest-Q1 \
  --tensor-parallel-size 8 \
  --reasoning-parser iquest_q1 \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_q1
```

To enable recursive MTP, use:

```bash
MODEL_ROOT="$(hf download IQuestLab/IQuest-Q1 --quiet)" && \
vllm serve "$MODEL_ROOT" \
  --served-model-name IQuest-Q1 \
  --tensor-parallel-size 8 \
  --reasoning-parser iquest_q1 \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_q1 \
  --enable-prefix-caching \
  --speculative-config '{
    "method": "eagle",
    "model": "'"$MODEL_ROOT"'/mtp",
    "num_speculative_tokens": 5,
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "standard",
    "enforce_eager": false
  }'
```

For an FP8 KV cache, add `--kv-cache-dtype fp8` (or `fp8_e4m3`). This
requires a device and FlashAttention version that support FP8 KV caching in
vLLM, such as Hopper with FlashAttention 3 and the default BF16 model dtype.
Add `--calculate-kv-scales` when
the checkpoint does not provide calibrated scales. The learned sink key stays
in the model dtype, and its attention score uses the dequantized query.
Both per-tensor and per-KV-head quantization scales are supported.

## Development

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```
