"""Base output parser class."""

from string import Formatter
from typing import TYPE_CHECKING, Any, Optional

from llama_index.core.output_parsers.base import ChainableOutputParser

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import (
        BaseOutputParser as LCOutputParser,
    )


class LangchainOutputParser(ChainableOutputParser):
    """Langchain output parser."""

    def __init__(
        self, output_parser: "LCOutputParser", format_key: Optional[str] = None
    ) -> None:
        """Init params."""
        self._output_parser = output_parser
        self._format_key = format_key

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # TODO: this object may be stringified by our upstream llmpredictor,
        # figure out better
        # ways to "convert" the object to a proper string format.
        return self._output_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        format_instructions = self._output_parser.get_format_instructions()

        # Optimization: Do not instantiate Formatter or parse unless "{" or "}" is in query
        if "{" in query or "}" in query:
            # Fast path: simple scan for '{' or '}' avoids unnecessary object creation
            # Only parse if needed
            # Optimization: Use a generator expression with next() early exit for any template variable in query
            for _, v, _, _ in Formatter().parse(query):
                if v is not None:
                    format_instructions = format_instructions.replace("{", "{{")
                    format_instructions = format_instructions.replace("}", "}}")
                    break


        if self._format_key is not None:
            fmt_query = query.format(**{self._format_key: format_instructions})
        else:
            fmt_query = query + "\n\n" + format_instructions

        return fmt_query
