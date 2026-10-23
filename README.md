# vllm-iquest-q1

A vLLM plugin for IQuest-Q1, with recursive MTP, reasoning parsing, and tool calling.

## Installation

Install the published package, which pins vLLM 0.30.0:

```bash
uv pip install vllm-iquest-q1==0.1.1
```

For development, install from this repository:

```bash
uv pip install .
```

The CUDA 13.0 image includes the same release wheel and vLLM 0.30.0:

```bash
docker pull iquestlabworkspace/vllm-iquest-q1:0.1.1-cu130
# The cu130 tag follows the current CUDA 13.0 release.
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
The supported vLLM version uses unit scales when the checkpoint does not
provide calibrated scales; it does not accept `--calculate-kv-scales`. Validate
generation quality with your checkpoint and workload. The learned sink key stays
in the model dtype, and its attention score uses the dequantized query.
Both per-tensor and per-KV-head quantization scales are supported.

## Development

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```

## Publishing

Build the wheel from a clean release commit, install that wheel in the validation
environment, and run the tests and real-model checks before publishing. Attach
the tested wheel and its `SHA256SUMS` file to a GitHub release whose tag matches
the package version (`v0.1.1`). Publishing the release triggers the existing PyPI
trusted publisher; the workflow verifies and uploads the attached wheel without
rebuilding it.

The Dockerfile reuses the previous CUDA 13.0 runtime by immutable digest. Stage
the tested plugin wheel, the upstream vLLM 0.30.0 Linux x86-64 wheel, and its required
`humming-kernels==0.1.12` x86-64 wheel in `dist/`. The build verifies their hashes
and installs these exact files without downloading or changing the base image's
other dependencies. Supply the plugin wheel's
SHA-256 and the source commit:

```bash
docker buildx build --platform linux/amd64 \
  --provenance=false --sbom=false \
  --build-arg IQUEST_WHEEL_SHA256="$(sha256sum dist/vllm_iquest_q1-0.1.1-py3-none-any.whl | cut -d ' ' -f 1)" \
  --build-arg IQUEST_PLUGIN_COMMIT="$(git rev-parse HEAD)" \
  -t iquestlabworkspace/vllm-iquest-q1:0.1.1-cu130 --push .
```

Verify the published manifest digest before promoting the release to `cu130`.
