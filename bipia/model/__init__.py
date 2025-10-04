# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
from pathlib import Path
from collections import OrderedDict
import logging

from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem

# Import vLLM-based models lazily/optionally so the package can be installed
# without GPU-specific dependencies (vllm, deepspeed).
try:
    from .llama import (
        Alpaca,
        Vicuna,
        Baize,
        StableVicuna,
        Koala,
        GPT4ALL,
        Wizard,
        Guanaco,
        Llama2,
    )
    from .vllm_worker import Dolly, StableLM, MPT, Mistral
    _VLLM_AVAILABLE = True
except Exception:  # ImportError and any runtime import issues from vllm
    Alpaca = Vicuna = Baize = StableVicuna = Koala = GPT4ALL = Wizard = Guanaco = Llama2 = None
    Dolly = StableLM = MPT = Mistral = None
    _VLLM_AVAILABLE = False
from .llm_worker import RwkvModel, OASST, ChatGLM, FastChatT5

logger = logging.getLogger(__name__)

LLM_NAME_TO_CLASS = OrderedDict(
    [
        ("gpt35", GPT35),
        ("gpt4", GPT4),
        ("gpt35_wosys", GPT35WOSystem),
        ("gpt4_wosys", GPT4WOSystem),
        ("rwkv", RwkvModel),
        ("oasst", OASST),
        ("chatglm", ChatGLM),
        ("t5", FastChatT5),
    ]
)

# Register vLLM-based models only when available
if _VLLM_AVAILABLE:
    LLM_NAME_TO_CLASS.update(
        OrderedDict(
            [
                ("alpaca", Alpaca),
                ("vicuna", Vicuna),
                ("baize", Baize),
                ("stablelm", StableLM),
                ("stablevicuna", StableVicuna),
                ("dolly", Dolly),
                ("koala", Koala),
                ("mpt", MPT),
                ("gpt4all", GPT4ALL),
                ("wizard", Wizard),
                ("guanaco", Guanaco),
                ("llama2", Llama2),
                ("mistral", Mistral),
            ]
        )
    )
else:
    logger.warning(
        "vLLM-based models are unavailable (vllm not installed). Install with 'pip install .[gpu]' to enable."
    )


class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            name = name
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )

        logger.info(f"Load {name} from name.")

        llm_cls = LLM_NAME_TO_CLASS[name]
        return llm_cls
