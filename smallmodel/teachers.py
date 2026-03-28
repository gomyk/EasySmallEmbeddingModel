"""Teacher model registry.

Pre-registered teachers and a function to add custom ones.
"""

from __future__ import annotations

import copy
from typing import Any

# ── Built-in Teacher Registry ───────────────────────────────────

TEACHERS: dict[str, dict[str, Any]] = {
    "minilm": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short_name": "MiniLM-L12",
        "hidden_dim": 384,
        "num_layers": 12,
        "intermediate_size": 1536,
        "vocab_size": 250002,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "modernbert": {
        "model_id": "answerdotai/ModernBERT-base",
        "short_name": "ModernBERT",
        "hidden_dim": 768,
        "num_layers": 22,
        "intermediate_size": 1152,
        "vocab_size": 50368,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": False,
    },
    "gte": {
        "model_id": "alibaba-NLP/gte-multilingual-base",
        "short_name": "GTE-multilingual",
        "hidden_dim": 768,
        "num_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 250048,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": True,
    },
    "me5": {
        "model_id": "intfloat/multilingual-e5-base",
        "short_name": "mE5-base",
        "hidden_dim": 768,
        "num_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 250002,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "me5s": {
        "model_id": "intfloat/multilingual-e5-small",
        "short_name": "mE5-small",
        "hidden_dim": 384,
        "num_layers": 12,
        "intermediate_size": 1536,
        "vocab_size": 250037,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "gemma_emb": {
        "model_id": "google/embeddinggemma-300m",
        "short_name": "EmbeddingGemma-300M",
        "hidden_dim": 768,
        "num_layers": 24,
        "intermediate_size": 1152,
        "vocab_size": 262144,
        "layer_accessor": "layers",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
        "num_attention_heads": 3,
        "num_kv_heads": 1,
        "head_dim": 256,
        "has_glu": True,
        "is_decoder": True,
        "license": "gemma",
        "license_notice": (
            "This model is a derivative of Google's Gemma. "
            "Gemma is provided under and subject to the Gemma Terms of Use "
            "found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). "
            "Use of this model must comply with the "
            "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."
        ),
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B",
        "short_name": "Qwen3-0.6B",
        "hidden_dim": 1024,
        "num_layers": 28,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": False,
        "num_attention_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "has_glu": True,
        "is_decoder": True,
    },
}


def register_teacher(key: str, *, model_id: str, short_name: str | None = None,
                     hidden_dim: int, num_layers: int, intermediate_size: int,
                     vocab_size: int, layer_accessor: str = "encoder.layer",
                     tokenizer_type: str = "unigram", trust_remote_code: bool = False,
                     **kwargs) -> dict[str, Any]:
    """Register a custom teacher model.

    Example::

        register_teacher(
            "my-bert",
            model_id="my-org/my-bert-base",
            short_name="MyBERT",
            hidden_dim=768, num_layers=12,
            intermediate_size=3072, vocab_size=30522,
        )
    """
    entry = {
        "model_id": model_id,
        "short_name": short_name or key,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "layer_accessor": layer_accessor,
        "tokenizer_type": tokenizer_type,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    TEACHERS[key] = entry
    return entry


def get_teacher(key: str) -> dict[str, Any]:
    """Get a teacher config by key. Raises KeyError if not found."""
    if key not in TEACHERS:
        raise KeyError(
            f"Unknown teacher '{key}'. Available: {sorted(TEACHERS.keys())}"
        )
    return copy.deepcopy(TEACHERS[key])
