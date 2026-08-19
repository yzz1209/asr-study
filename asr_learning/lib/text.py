from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(
    r"[\s"
    r"\u3000-\u303F"  # CJK punctuation
    r"\uFF00-\uFFEF"  # fullwidth forms
    r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    r"]+"
)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_punct_and_space(text: str) -> str:
    text = nfkc(text)
    return _PUNCT_RE.sub("", text)


def normalize_for_cer(text: str, *, keep_punct: bool = False) -> str:
    """Chinese subtitle CER usually ignores whitespace; punct is optional."""
    text = nfkc(text).replace("\r\n", "\n").replace("\r", "\n")
    if keep_punct:
        return "".join(ch for ch in text if not ch.isspace())
    return strip_punct_and_space(text)


def char_ngrams(text: str, n: int) -> list[str]:
    if n <= 0 or len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]
