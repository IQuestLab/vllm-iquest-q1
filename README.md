# vllm-iquest-q1

A vLLM plugin for IQuest-Q1, with recursive MTP, reasoning parsing, and tool calling.

## Installation

Install the release from PyPI:

```bash
pip install vllm-iquest-q1
```

The release pins `vllm==0.30.0`. Use a Linux environment supported by that
vLLM release, with a compatible NVIDIA driver for GPU inference. The plugin
wheel contains Python code; pip installs vLLM and its dependencies separately.
Model weights are downloaded separately.

To install from this repository:

```bash
pip install .
```

The original development environment used vLLM commit
`81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`. To keep an existing compatible
vLLM environment instead of resolving the release dependencies:

```bash
uv pip install --no-deps .
```

Or use Docker:
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

## Development

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```

## Publishing

Releases use PyPI Trusted Publishing. Configure the PyPI project
`vllm-iquest-q1` to trust the GitHub repository `IQuestLab/vllm-iquest-q1`,
workflow `publish.yml`, and environment `pypi`.

Build and test the wheel before creating a GitHub release. Attach the tested
wheel and a `SHA256SUMS` file containing its SHA-256 and filename to a draft
release tagged `v<package-version>`, then publish the release. The tag must
point to the commit used to build and test the wheel.

The workflow verifies the checksum, package name, and version, then uploads
the existing wheel to PyPI. It does not rebuild the package.
