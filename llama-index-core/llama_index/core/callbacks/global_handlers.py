from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.simple_llm_handler import SimpleLLMHandler

_EVAL_HANDLERS = {
    "wandb": (
        "llama_index.callbacks.wandb",
        "WandbCallbackHandler",
        "WandbCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-wandb`"
    ),
    "openinference": (
        "llama_index.callbacks.openinference",
        "OpenInferenceCallbackHandler",
        "OpenInferenceCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-openinference`"
    ),
    "arize_phoenix": (
        "llama_index.callbacks.arize_phoenix",
        "arize_phoenix_callback_handler",
        "ArizePhoenixCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-arize-phoenix`"
    ),
    "honeyhive": (
        "llama_index.callbacks.honeyhive",
        "honeyhive_callback_handler",
        "HoneyHiveCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-honeyhive`"
    ),
    "promptlayer": (
        "llama_index.callbacks.promptlayer",
        "PromptLayerHandler",
        "PromptLayerHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-promptlayer`"
    ),
    "deepeval": (
        "llama_index.callbacks.deepeval",
        "deepeval_callback_handler",
        "DeepEvalCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-deepeval`"
    ),
    "argilla": (
        "llama_index.callbacks.argilla",
        "argilla_callback_handler",
        "ArgillaCallbackHandler is not installed. "
        "Please install it using `pip install llama-index-callbacks-argilla`"
    ),
}


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""
    import llama_index.core

    llama_index.core.global_handler = create_global_handler(eval_mode, **eval_params)


def create_global_handler(eval_mode: str, **eval_params: Any) -> BaseCallbackHandler:
    """Get global eval handler."""
    if eval_mode == "simple":
        handler = SimpleLLMHandler(**eval_params)
    elif eval_mode in _EVAL_HANDLERS:
        module_name, attr_name, error_msg = _EVAL_HANDLERS[eval_mode]
        try:
            module = __import__(module_name, fromlist=[attr_name])
            handler_cls = getattr(module, attr_name)
        except ImportError:
            raise ImportError(error_msg)
        handler = handler_cls(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler
