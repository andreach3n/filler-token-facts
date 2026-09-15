"""Token counting with each study model's own tokenizer.

Opus 4.5's tokenizer isn't public; its counts come from the Anthropic count_tokens endpoint
when the API runs are built.
"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

import fst  # noqa: F401  (sets SSL_CERT_FILE before any download)

TOKENIZER_REPOS = {
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
}


@lru_cache(maxsize=None)
def get_tokenizer(model: str) -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER_REPOS[model])


def n_tokens(text: str, model: str) -> int:
    return len(get_tokenizer(model).encode(text, add_special_tokens=False).ids)


def n_tokens_after(prefix: str, text: str, model: str) -> int:
    """Tokens `text` adds when it follows `prefix`, so boundary merges are counted as in the prompt."""
    return n_tokens(prefix + text, model) - n_tokens(prefix, model)
