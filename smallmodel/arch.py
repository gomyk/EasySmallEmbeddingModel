"""Architecture utilities: layer pruning, hidden dim reduction, vocab pruning."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


# ── Layer Access ─────────────────────────────────────────────────

def get_layers(model, layer_accessor: str):
    obj = model
    for attr in layer_accessor.split("."):
        obj = getattr(obj, attr)
    return obj


def set_layers(model, layer_accessor: str, new_layers):
    parts = layer_accessor.split(".")
    obj = model
    for attr in parts[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], new_layers)


def discover_layer_accessor(model) -> str:
    candidates = [
        "encoder.layer", "encoder.layers", "layers",
        "transformer.layer", "transformer.layers",
    ]
    for path in candidates:
        try:
            layers = get_layers(model, path)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return path
        except AttributeError:
            continue
    raise ValueError(f"Could not detect layer accessor for {type(model).__name__}")


# ── Layer Pruning ────────────────────────────────────────────────

def prune_layers(model, layer_indices: list[int], layer_accessor: str | None = None):
    if layer_accessor is None:
        layer_accessor = discover_layer_accessor(model)
    layers = get_layers(model, layer_accessor)
    kept = nn.ModuleList([layers[i] for i in layer_indices])
    set_layers(model, layer_accessor, kept)
    model.config.num_hidden_layers = len(layer_indices)
    return model


def create_pruned_student(teacher_model_id: str, layer_indices: list[int],
                          layer_accessor: str | None = None,
                          trust_remote_code: bool = False):
    """Load teacher and prune layers to create a student.

    Returns (student_model, tokenizer).
    """
    model = AutoModel.from_pretrained(teacher_model_id, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=trust_remote_code)

    if layer_accessor is None:
        layer_accessor = discover_layer_accessor(model)

    student = copy.deepcopy(model)
    student = prune_layers(student, layer_indices, layer_accessor)
    return student, tokenizer


# ── Hidden Dimension Reduction ───────────────────────────────────

def reduce_hidden_dim(model, new_hidden_dim: int, new_intermediate_size: int | None = None,
                      trust_remote_code: bool = False):
    old_hidden = model.config.hidden_size
    if new_hidden_dim >= old_hidden:
        return model

    old_inter = getattr(model.config, 'intermediate_size', old_hidden * 4)
    if new_intermediate_size is None:
        ratio = new_hidden_dim / old_hidden
        new_intermediate_size = max(64, (int(old_inter * ratio) // 64) * 64)

    new_config = copy.deepcopy(model.config)
    new_config.hidden_size = new_hidden_dim
    new_config.intermediate_size = new_intermediate_size

    ratio = new_hidden_dim / old_hidden
    old_n_kv = getattr(new_config, 'num_key_value_heads', None)

    if hasattr(new_config, 'num_attention_heads'):
        n_heads = getattr(new_config, 'num_attention_heads')
        if n_heads is not None:
            n_heads = max(1, int(n_heads * ratio))
            while new_hidden_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            new_config.num_attention_heads = n_heads

    if old_n_kv is not None and hasattr(new_config, 'num_key_value_heads'):
        n_heads = getattr(new_config, 'num_attention_heads', n_heads)
        n_kv = max(1, int(old_n_kv * ratio))
        while n_kv > 1 and (n_heads % n_kv != 0 or new_hidden_dim % n_kv != 0):
            n_kv -= 1
        new_config.num_key_value_heads = n_kv

    if hasattr(new_config, 'head_dim') and new_config.head_dim is not None:
        new_heads = getattr(new_config, 'num_attention_heads', 1)
        new_config.head_dim = new_hidden_dim // new_heads

    new_model = AutoModel.from_config(new_config, trust_remote_code=trust_remote_code)

    old_sd = model.state_dict()
    new_sd = new_model.state_dict()

    for key in new_sd:
        if key not in old_sd:
            continue
        old_t = old_sd[key]
        new_t = new_sd[key]
        if old_t.shape == new_t.shape:
            new_sd[key] = old_t.clone()
        else:
            slices = tuple(
                slice(0, min(s_new, s_old))
                for s_new, s_old in zip(new_t.shape, old_t.shape)
            )
            sliced = old_t[slices]
            if sliced.shape == new_t.shape:
                new_sd[key] = sliced.clone()
            else:
                target_slices = tuple(slice(0, s) for s in sliced.shape)
                new_sd[key][target_slices] = sliced.clone()

    new_model.load_state_dict(new_sd)

    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id", "mask_token_id"]:
        old_val = getattr(model.config, attr, None)
        if old_val is not None:
            setattr(new_model.config, attr, old_val)

    return new_model


# ── Tokenizer Type Detection ────────────────────────────────────

def detect_tokenizer_type(tokenizer) -> str:
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    return tok_json["model"]["type"]


# ── Vocab Pruning ────────────────────────────────────────────────

def collect_corpus_tokens(tokenizer, texts: list[str] | None = None,
                          max_vocab: int | None = None,
                          vocab_keep_ratio: float | None = None) -> list[int]:
    """Collect tokens used in a corpus, returning sorted IDs to keep."""
    if texts is None:
        texts = _get_default_multilingual_samples()

    freq = Counter()
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=True, max_length=128)
        for ids in encoded["input_ids"]:
            freq.update(ids)

    keep_ids = set(tokenizer.all_special_ids)

    basic_chars = list("0123456789.,!?;:'\"-()[]{}/@#$%^&*+=<>~_ \t\n")
    for ch in basic_chars:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        keep_ids.update(ids)

    tok_type = detect_tokenizer_type(tokenizer)
    if tok_type == "BPE":
        tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
        vocab = tok_json["model"]["vocab"]
        for token, tid in vocab.items():
            if tid < 256 or len(token) <= 1:
                keep_ids.add(tid)

    if vocab_keep_ratio is not None:
        total_freq = sum(freq.values())
        target_freq = total_freq * vocab_keep_ratio
        corpus_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
        cumsum = 0
        for tid in corpus_tokens:
            keep_ids.add(tid)
            cumsum += freq[tid]
            if cumsum >= target_freq:
                break
    elif max_vocab is not None:
        remaining = max_vocab - len(keep_ids)
        if remaining > 0:
            for tid, _ in freq.most_common():
                if tid not in keep_ids:
                    keep_ids.add(tid)
                    if len(keep_ids) >= max_vocab:
                        break
    else:
        keep_ids.update(freq.keys())

    return sorted(keep_ids)


def prune_tokenizer_and_embeddings(model, tokenizer, keep_ids: list[int], save_dir: str):
    """Prune tokenizer vocab and model embeddings simultaneously."""
    os.makedirs(save_dir, exist_ok=True)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    model_type = tok_json["model"]["type"]

    if model_type == "Unigram":
        tok_json = _prune_unigram(tok_json, keep_ids, old_to_new)
    elif model_type == "BPE":
        tok_json = _prune_bpe(tok_json, keep_ids, old_to_new)
    elif model_type == "WordPiece":
        tok_json = _prune_wordpiece(tok_json, keep_ids, old_to_new)
    else:
        tokenizer.save_pretrained(save_dir)
        model = _prune_embeddings(model, keep_ids)
        return model

    if "added_tokens" in tok_json:
        new_added = []
        for at in tok_json["added_tokens"]:
            old_id = at["id"]
            if old_id in old_to_new:
                at["id"] = old_to_new[old_id]
                new_added.append(at)
        tok_json["added_tokens"] = new_added

    pp = tok_json.get("post_processor")
    if pp and "special_tokens" in pp:
        for token_name, token_info in pp["special_tokens"].items():
            if "ids" in token_info:
                token_info["ids"] = [
                    old_to_new[oid] for oid in token_info["ids"] if oid in old_to_new
                ]

    tokenizer.save_pretrained(save_dir)

    tok_json_path = os.path.join(save_dir, "tokenizer.json")
    with open(tok_json_path, "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False)

    added_tokens_path = os.path.join(save_dir, "added_tokens.json")
    if os.path.exists(added_tokens_path):
        with open(added_tokens_path, "r", encoding="utf-8") as f:
            added_tokens = json.load(f)
        new_added_tokens = {}
        for token_str, old_id in added_tokens.items():
            if old_id in old_to_new:
                new_added_tokens[token_str] = old_to_new[old_id]
        with open(added_tokens_path, "w", encoding="utf-8") as f:
            json.dump(new_added_tokens, f, ensure_ascii=False)

    tok_config_path = os.path.join(save_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path, "r", encoding="utf-8") as f:
            tok_config = json.load(f)
        if "added_tokens_decoder" in tok_config:
            new_decoder = {}
            for old_id_str, token_info in tok_config["added_tokens_decoder"].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_decoder[str(old_to_new[old_id])] = token_info
            tok_config["added_tokens_decoder"] = new_decoder
        with open(tok_config_path, "w", encoding="utf-8") as f:
            json.dump(tok_config, f, ensure_ascii=False, indent=2)

    model = _prune_embeddings(model, keep_ids)
    return model


def save_as_sentence_transformer(model, tokenizer, save_path: str):
    """Save HF model as SentenceTransformer format."""
    from sentence_transformers import SentenceTransformer, models as st_models
    import shutil
    import glob as _glob

    hf_tmp = os.path.join(save_path, "_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    model.save_pretrained(hf_tmp)
    tokenizer.save_pretrained(hf_tmp)

    config_path = os.path.join(hf_tmp, "config.json")
    is_custom = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        is_custom = "auto_map" in config

    if is_custom:
        _copy_custom_code_files(model, hf_tmp)
    else:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config.pop("_name_or_path", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    word_model = st_models.Transformer(
        hf_tmp,
        config_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )
    pool_model = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    st_model = SentenceTransformer(modules=[word_model, pool_model])
    st_model.save(save_path)
    shutil.rmtree(hf_tmp, ignore_errors=True)
    return st_model


# ── Internal helpers ─────────────────────────────────────────────

def _prune_unigram(tok_json, keep_ids, old_to_new):
    old_vocab = tok_json["model"]["vocab"]
    new_vocab = []
    for old_id in keep_ids:
        if old_id < len(old_vocab):
            new_vocab.append(old_vocab[old_id])
    tok_json["model"]["vocab"] = new_vocab
    old_unk_id = tok_json["model"].get("unk_id")
    if old_unk_id is not None and old_unk_id in old_to_new:
        tok_json["model"]["unk_id"] = old_to_new[old_unk_id]
    return tok_json


def _prune_bpe(tok_json, keep_ids, old_to_new):
    old_vocab = tok_json["model"]["vocab"]
    keep_ids_set = set(keep_ids)
    new_vocab = {}
    for token, old_id in old_vocab.items():
        if old_id in keep_ids_set:
            new_vocab[token] = old_to_new[old_id]
    tok_json["model"]["vocab"] = new_vocab
    kept_tokens = set(new_vocab.keys())
    if "merges" in tok_json["model"]:
        new_merges = []
        for merge in tok_json["model"]["merges"]:
            parts = merge if isinstance(merge, list) else merge.split(" ")
            if len(parts) == 2:
                merged_token = parts[0] + parts[1]
                if (parts[0] in kept_tokens and parts[1] in kept_tokens
                        and merged_token in kept_tokens):
                    new_merges.append(merge)
        tok_json["model"]["merges"] = new_merges
    return tok_json


def _prune_wordpiece(tok_json, keep_ids, old_to_new):
    old_vocab = tok_json["model"]["vocab"]
    keep_ids_set = set(keep_ids)
    new_vocab = {}
    for token, old_id in old_vocab.items():
        if old_id in keep_ids_set:
            new_vocab[token] = old_to_new[old_id]
    tok_json["model"]["vocab"] = new_vocab
    return tok_json


def _prune_embeddings(model, keep_ids):
    old_emb = model.get_input_embeddings()
    old_weight = old_emb.weight.data
    new_vocab_size = len(keep_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id",
                 "mask_token_id", "decoder_start_token_id"]:
        old_id = getattr(model.config, attr, None)
        if old_id is not None:
            if old_id in old_to_new:
                setattr(model.config, attr, old_to_new[old_id])
            else:
                setattr(model.config, attr, None)

    padding_idx = getattr(old_emb, 'padding_idx', None)
    if padding_idx is not None:
        padding_idx = old_to_new.get(padding_idx, None)
    new_emb = nn.Embedding(new_vocab_size, old_weight.shape[1], padding_idx=padding_idx)
    for new_id, old_id in enumerate(keep_ids):
        if old_id < old_weight.shape[0]:
            new_emb.weight.data[new_id] = old_weight[old_id]

    model.set_input_embeddings(new_emb)
    model.config.vocab_size = new_vocab_size
    return model


def _copy_custom_code_files(model, target_dir):
    import shutil
    import glob as _glob

    source_path = getattr(model.config, '_name_or_path', None)
    if not source_path:
        return

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache_name = "models--" + source_path.replace("/", "--")
    model_cache_dir = os.path.join(cache_dir, model_cache_name)

    if os.path.exists(model_cache_dir):
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                latest = os.path.join(snapshots_dir, snapshots[-1])
                for py_file in _glob.glob(os.path.join(latest, "*.py")):
                    fname = os.path.basename(py_file)
                    dest = os.path.join(target_dir, fname)
                    if not os.path.exists(dest):
                        shutil.copy2(py_file, dest)


def _get_default_multilingual_samples() -> list[str]:
    return [
        "예약 좀 해줘", "지난번 주문 뭐였지?", "안녕하세요 반갑습니다",
        "Book a table for me", "What did I order last time?", "Hello how are you",
        "予約をお願いします", "前回の注文は何でしたか", "こんにちは元気ですか",
        "帮我预约一下", "上次我点了什么", "你好你好吗",
        "Reserva una mesa", "Qué pedí la última vez", "Hola cómo estás",
        "Réservez une table", "Qu'est-ce que j'ai commandé", "Bonjour comment allez-vous",
        "Reservieren Sie einen Tisch", "Was habe ich bestellt", "Hallo wie geht es",
    ] * 10
