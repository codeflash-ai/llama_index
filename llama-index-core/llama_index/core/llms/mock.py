from typing import Any, Callable, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import PydanticProgramMode


class MockLLM(CustomLLM):
    max_tokens: Optional[int]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    def _generate_text(self, length: int) -> str:
        # Generate the repeated sentence as a single string for better performance
        # Fast path for common case: avoid join+list
        if length <= 0:
            return ""
        if length == 1:
            return "text"
        # The string "text " has length 5 except for the last word
        return "text" + (" text" * (length - 1))

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response_text = (
            self._generate_text(self.max_tokens) if self.max_tokens else prompt
        )

        return CompletionResponse(
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen_prompt() -> CompletionResponseGen:
            for ch in prompt:
                yield CompletionResponse(
                    text=prompt,
                    delta=ch,
                )

        def gen_response(max_tokens: int) -> CompletionResponseGen:
            for i in range(max_tokens):
                response_text = self._generate_text(i)
                yield CompletionResponse(
                    text=response_text,
                    delta="text ",
                )

        return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()
