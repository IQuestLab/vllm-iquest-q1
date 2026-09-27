# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_registered = False


def register() -> None:
    """Register configs, lazy model classes, and the existing parser interfaces."""
    global _registered
    if _registered:
        return

    from transformers import AutoConfig

    from vllm import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    from .configs import IQuestQ1Config, IQuestQ1MTPConfig

    for config in (IQuestQ1Config, IQuestQ1MTPConfig):
        AutoConfig.register(config.model_type, config, exist_ok=True)

    ModelRegistry.register_model(
        "IQuestQ1ForCausalLM", "vllm_iquest_q1.model:IQuestQ1ForCausalLM"
    )
    for architecture in (
        "IQuestQ1MTP",
        "EagleIQuestQ1MTP",
        "IQuestQ1MtpRecursive",
        "IQuestQ1MTPRecursive",
        "EagleIQuestQ1MtpRecursive",
        "EagleIQuestQ1MTPRecursive",
    ):
        ModelRegistry.register_model(
            architecture, "vllm_iquest_q1.mtp_recursive:IQuestQ1MTPRecursive"
        )
    ReasoningParserManager.register_lazy_module(
        "iquest_q1", "vllm_iquest_q1.reasoning_parser", "IQuestQ1ReasoningParser"
    )
    ToolParserManager.register_lazy_module(
        "iquest_q1", "vllm_iquest_q1.tool_parser", "IQuestQ1ToolParser"
    )
    _registered = True
