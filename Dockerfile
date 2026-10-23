# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0

# Preserve the CUDA 13.0 runtime and its matching PyTorch/FlashInfer/CUTLASS stack.
FROM iquestlabworkspace/vllm-iquest-q1@sha256:c1685cbf33e3f13bccc0c92729618705fa7a7b0cef0996de04bdbe6d2c3f72e3

ARG TARGETARCH
ARG IQUEST_PLUGIN_COMMIT
ARG IQUEST_WHEEL_SHA256

LABEL org.opencontainers.image.title="vLLM with the IQuest-Q1 plugin" \
      org.opencontainers.image.source="https://github.com/IQuestLab/vllm-iquest-q1" \
      org.opencontainers.image.version="0.1.1" \
      org.opencontainers.image.revision="$IQUEST_PLUGIN_COMMIT" \
      ai.vllm.source.commit="" \
      ai.vllm.version="0.30.0" \
      ai.iquest.plugin.commit="$IQUEST_PLUGIN_COMMIT"

WORKDIR /opt/vllm

# Bind-mounted wheels are verified and installed without retaining them in a layer.
RUN --mount=type=bind,source=dist,target=/opt/wheels \
    test "$TARGETARCH" = amd64 \
    && test -n "$IQUEST_PLUGIN_COMMIT" \
    && test -n "$IQUEST_WHEEL_SHA256" \
    && cd /opt/wheels \
    && echo "ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4  vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl" | sha256sum --check --strict \
    && echo "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331  humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl" | sha256sum --check --strict \
    && echo "$IQUEST_WHEEL_SHA256  vllm_iquest_q1-0.1.1-py3-none-any.whl" | sha256sum --check --strict \
    && uv pip install --python "$VIRTUAL_ENV/bin/python" --no-index --no-deps --no-cache \
        ./vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl \
        ./humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl \
        ./vllm_iquest_q1-0.1.1-py3-none-any.whl \
    && uv pip check --python "$VIRTUAL_ENV/bin/python"

# Triton compiles Python extensions at runtime, even with prebuilt wheels.
RUN "$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile

with tempfile.TemporaryDirectory(dir="/opt/vllm") as directory:
    source = Path(directory) / "_header_check.c"
    extension = source.with_name("_header_check" + sysconfig.get_config_var("EXT_SUFFIX"))
    source.write_text('''
#include <Python.h>
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_header_check", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit__header_check(void) { return PyModule_Create(&module); }
''')
    subprocess.run(
        shlex.split(sysconfig.get_config_var("LDSHARED"))
        + ["-fPIC", "-I" + sysconfig.get_path("include"),
           str(source), "-o", str(extension)],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("_header_check", extension)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
print("Python C extension compile and import check passed")
PY

# CPU-only build check; this does not override runtime device selection.
RUN VLLM_TARGET_DEVICE=cpu "$VIRTUAL_ENV/bin/python" - <<'PY'
from dataclasses import fields
from importlib.metadata import version

from transformers import AutoConfig
from vllm.config import SpeculativeConfig
from vllm_iquest_q1 import register

register()
assert AutoConfig.for_model("iquest_q1").model_type == "iquest_q1"
assert AutoConfig.for_model("iquest_q1_mtp").model_type == "iquest_q1_mtp"
assert version("vllm-iquest-q1") == "0.1.1"
assert version("vllm") == "0.30.0"
assert "draft_sample_method" in {field.name for field in fields(SpeculativeConfig)}
print({name: version(name) for name in ("vllm", "vllm-iquest-q1", "instanttensor")})
PY

EXPOSE 8000
ENTRYPOINT ["/opt/vllm/.venv/bin/vllm", "serve"]
CMD ["--help"]
