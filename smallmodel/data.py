"""Distillation corpus loading utilities."""

from __future__ import annotations

import os


def load_distill_corpus(max_per_lang: int = 5000,
                        cache_dir: str = "data/distill_corpus") -> list[str]:
    """Collect texts from MTEB task datasets for distillation/vocab analysis."""
    from datasets import load_dataset

    cache_file = os.path.join(cache_dir, f"distill_texts_{max_per_lang}.txt")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded cached corpus: {len(texts):,} sentences")
        return texts

    os.makedirs(cache_dir, exist_ok=True)
    texts = []

    MASSIVE_LANGS = {
        "ko": "ko-KR", "en": "en-US", "ja": "ja-JP", "zh": "zh-CN",
        "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "pt": "pt-PT",
        "it": "it-IT", "ru": "ru-RU", "ar": "ar-SA", "hi": "hi-IN",
        "th": "th-TH", "vi": "vi-VN", "id": "id-ID", "tr": "tr-TR",
        "nl": "nl-NL", "pl": "pl-PL",
    }

    print("  Loading MASSIVE dataset...")
    for lang, subset in MASSIVE_LANGS.items():
        try:
            try:
                ds = load_dataset("mteb/amazon_massive_intent", subset, split="train")
            except Exception:
                ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
            lang_texts = [row["text"] for row in ds if row.get("text")][:max_per_lang]
            texts.extend(lang_texts)
            print(f"    {lang}: {len(lang_texts)} sentences")
        except Exception as e:
            print(f"    {lang}: failed - {e}")

    print("  Loading STS benchmark...")
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="train")
        for row in ds:
            for field in ["sentence1", "sentence2"]:
                if row.get(field):
                    texts.append(row[field])
    except Exception as e:
        print(f"    STSBenchmark: failed - {e}")

    print("  Loading Banking77...")
    try:
        ds = load_dataset("mteb/banking77", split="train")
        b77_texts = [row["text"] for row in ds if row.get("text")]
        texts.extend(b77_texts)
    except Exception as e:
        print(f"    Banking77: failed - {e}")

    with open(cache_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n")

    print(f"  Total corpus: {len(texts):,} sentences")
    return texts


def load_mteb_task_texts(max_per_dataset: int = 10000,
                         cache_dir: str = "data/distill_corpus") -> list[str]:
    """Load MTEB Classification/Clustering/STS task texts for distillation."""
    from datasets import load_dataset

    DISTILL_DATASETS = {
        "amazon_counterfactual": {"hf_id": "mteb/amazon_counterfactual", "text_fields": ["text"], "splits": ["train", "test"]},
        "banking77": {"hf_id": "mteb/banking77", "text_fields": ["text"], "splits": ["train", "test"]},
        "imdb": {"hf_id": "mteb/imdb", "text_fields": ["text"], "splits": ["train", "test"]},
        "mtop_domain": {"hf_id": "mteb/mtop_domain", "text_fields": ["text"], "splits": ["train", "test"]},
        "massive_intent": {"hf_id": "mteb/amazon_massive_intent", "text_fields": ["text"], "splits": ["train"], "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"]},
        "massive_scenario": {"hf_id": "mteb/amazon_massive_scenario", "text_fields": ["text"], "splits": ["train"], "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"]},
        "toxic_conversations": {"hf_id": "mteb/toxic_conversations_50k", "text_fields": ["text"], "splits": ["test"]},
        "tweet_sentiment": {"hf_id": "mteb/tweet_sentiment_extraction", "text_fields": ["text"], "splits": ["train", "test"]},
        "stsb": {"hf_id": "mteb/stsbenchmark-sts", "text_fields": ["sentence1", "sentence2"], "splits": ["train", "test"]},
        "sickr": {"hf_id": "mteb/SICK-R", "text_fields": ["sentence1", "sentence2"], "splits": ["test"]},
    }

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"mteb_distill_{max_per_dataset}.txt")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded cached distillation corpus: {len(texts):,} sentences")
        return texts

    print("Loading MTEB task datasets for distillation...")
    all_texts = []

    for ds_name, ds_config in DISTILL_DATASETS.items():
        hf_id = ds_config["hf_id"]
        text_fields = ds_config["text_fields"]
        splits = ds_config["splits"]
        subsets = ds_config.get("subsets", [None])

        for subset in subsets:
            for split in splits:
                try:
                    ds = load_dataset(hf_id, subset, split=split) if subset else load_dataset(hf_id, split=split)
                    count = 0
                    for row in ds:
                        for field in text_fields:
                            text = row.get(field, "")
                            if text and len(str(text)) > 5:
                                all_texts.append(str(text))
                                count += 1
                                if count >= max_per_dataset:
                                    break
                        if count >= max_per_dataset:
                            break
                except Exception:
                    pass

    unique_texts = list(set(all_texts))
    with open(cache_file, "w", encoding="utf-8") as f:
        for text in unique_texts:
            f.write(text.strip() + "\n")

    print(f"  Total unique texts: {len(unique_texts):,}")
    return unique_texts
