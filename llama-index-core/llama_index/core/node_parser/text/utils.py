import logging
from typing import Callable, List

from llama_index.core.node_parser.interface import TextSplitter

logger = logging.getLogger(__name__)


def truncate_text(text: str, text_splitter: TextSplitter) -> str:
    """Truncate text to fit within the chunk size."""
    chunks = text_splitter.split_text(text)
    return chunks[0]


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text with separator and keep the separator at the end of each split."""
    if separator == "":
        # Behavior preservation: match str.split('')
        raise ValueError("empty separator")

    parts = text.split(separator)
    n = len(parts)
    if n == 1:
        # Only one split part; just return, possibly empty
        return [parts[0]] if parts[0] else []
    
    # Fast direct construction: allocate once, avoid extra list comprehensions
    result: List[str] = []
    # First part doesn't get separator in front
    if parts[0]:
        result.append(parts[0])
    # Remaining parts get separator prepended
    for s in parts[1:]:
        combined = separator + s
        if combined:
            result.append(combined)
    return result


def split_by_sep(sep: str, keep_sep: bool = True) -> Callable[[str], List[str]]:
    """Split text by separator."""
    if keep_sep:
        return lambda text: split_text_keep_separator(text, sep)
    else:
        return lambda text: text.split(sep)


def split_by_char() -> Callable[[str], List[str]]:
    """Split text by character."""
    return lambda text: list(text)


def split_by_sentence_tokenizer() -> Callable[[str], List[str]]:
    import nltk

    tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # get the spans and then return the sentences
    # using the start index of each span
    # instead of using end, use the start of the next span if available
    def split(text: str) -> List[str]:
        spans = list(tokenizer.span_tokenize(text))
        sentences = []
        for i, span in enumerate(spans):
            start = span[0]
            if i < len(spans) - 1:
                end = spans[i + 1][0]
            else:
                end = len(text)
            sentences.append(text[start:end])

        return sentences

    return split


def split_by_regex(regex: str) -> Callable[[str], List[str]]:
    """Split text by regex."""
    import re

    return lambda text: re.findall(regex, text)


def split_by_phrase_regex() -> Callable[[str], List[str]]:
    """Split text by phrase regex.

    This regular expression will split the sentences into phrases,
    where each phrase is a sequence of one or more non-comma,
    non-period, and non-semicolon characters, followed by an optional comma,
    period, or semicolon. The regular expression will also capture the
    delimiters themselves as separate items in the list of phrases.
    """
    regex = "[^,.;。]+[,.;。]?"
    return split_by_regex(regex)
