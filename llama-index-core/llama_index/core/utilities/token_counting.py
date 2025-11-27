# Modified from:
# https://github.com/nyno-ai/openai-token-counter

from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.utils import get_tokenizer


class TokenCounter:
    """Token counter class.

    Attributes:
        model (Optional[str]): The model to use for token counting.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None) -> None:
        self.tokenizer = tokenizer or get_tokenizer()

    def get_string_tokens(self, string: str) -> int:
        """Get the token count for a string.

        Args:
            string (str): The string to count.

        Returns:
            int: The token count.
        """
        return len(self.tokenizer(string))

    def estimate_tokens_in_messages(self, messages: List[ChatMessage]) -> int:
        """Estimate token count for a single message.

        Args:
            message (OpenAIMessage): The message to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        tokens = 0


        get_string_tokens = self.get_string_tokens  # Localize for performance

        # Avoid dict copy unless needed, cache function_call presence, and avoid repeated attribute access
        MessageRole_FUNCTION = MessageRole.FUNCTION

        for message in messages:
            role = message.role
            content = message.content
            additional_kwargs = message.additional_kwargs

            # Pre-compute per-message token additions and possible role
            if role:
                tokens += get_string_tokens(role)

            if content:
                tokens += get_string_tokens(content)

            # Fast path: avoid creating dictionary copy for most messages

            if "function_call" in additional_kwargs:
                function_call = additional_kwargs["function_call"]

                name = function_call.get("name")
                if name is not None:
                    tokens += get_string_tokens(name)

                arguments = function_call.get("arguments")
                if arguments is not None:
                    tokens += get_string_tokens(arguments)


                tokens += 3  # Additional tokens for function call

            tokens += 3  # Add three per message

            # Check for FUNCTION role and adjust
            if role == MessageRole_FUNCTION:
                tokens -= 2  # Subtract 2 if role is "function"

        return tokens

    def estimate_tokens_in_functions(self, functions: List[Dict[str, Any]]) -> int:
        """Estimate token count for the functions.

        We take here a list of functions created using the `to_openai_spec` function (or similar).

        Args:
            function (list[Dict[str, Any]]): The functions to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        prompt_definition = str(functions)
        tokens = self.get_string_tokens(prompt_definition)
        tokens += 9  # Additional tokens for function definition
        return tokens
