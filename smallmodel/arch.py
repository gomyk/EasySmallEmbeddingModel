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


# ── Attention Head Pruning ───────────────────────────────────────

def prune_attention_heads(model, num_heads_to_keep: int, layer_accessor: str | None = None):
    """Prune attention heads by importance (L1 norm of output weights).

    Keeps the hidden_dim unchanged. Only the internal attention projection
    dimension is reduced: Q/K/V go from [hidden, hidden] to [hidden, heads*head_dim],
    O goes from [hidden, hidden] to [heads*head_dim, hidden].

    Supports:
      - Fused QKV: qkv_proj [3*n_heads*head_dim, hidden] (GTE, etc.)
      - Separate Q/K/V: query/key/value [n_heads*head_dim, hidden] (BERT, XLM-R)
      - Decoder-style: q_proj/k_proj/v_proj (Qwen, Gemma, etc.)
    """
    if layer_accessor is None:
        layer_accessor = discover_layer_accessor(model)

    config = model.config
    old_num_heads = config.num_attention_heads
    head_dim = config.hidden_size // old_num_heads

    if num_heads_to_keep >= old_num_heads:
        return model

    layers = get_layers(model, layer_accessor)

    for layer_idx, layer in enumerate(layers):
        # Find attention module and weight names
        attn_info = _find_attention_weights(layer)
        if attn_info is None:
            print(f"  Warning: could not find attention weights in layer {layer_idx}")
            continue

        # Score heads by L1 norm of output projection
        o_weight = attn_info["o_weight"]  # [hidden, n_heads*head_dim] or transposed
        head_scores = _score_heads(o_weight, old_num_heads, head_dim)

        # Select top-k heads
        _, keep_indices = torch.topk(head_scores, num_heads_to_keep)
        keep_indices = keep_indices.sort().values

        # Prune Q, K, V, O
        _apply_head_pruning(attn_info, keep_indices, old_num_heads, head_dim)

    # Update config
    config.num_attention_heads = num_heads_to_keep
    # Store original head_dim so custom model code can use it
    config.head_dim = head_dim
    if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads is not None:
        # Scale KV heads proportionally
        old_kv = config.num_key_value_heads
        new_kv = max(1, int(old_kv * num_heads_to_keep / old_num_heads))
        while num_heads_to_keep % new_kv != 0 and new_kv > 1:
            new_kv -= 1
        config.num_key_value_heads = new_kv

    print(f"  Head pruning: {old_num_heads} -> {num_heads_to_keep} heads "
          f"(attn dim: {old_num_heads*head_dim} -> {num_heads_to_keep*head_dim})")
    return model


def _find_attention_weights(layer) -> dict | None:
    """Detect attention weight layout in a transformer layer."""
    result = {}

    # Try fused QKV (GTE style: attention.qkv_proj)
    for name, param in layer.named_parameters():
        lname = name.lower()
        if "qkv_proj" in lname or "qkv" in lname and "weight" in lname:
            result["qkv_weight"] = param
            result["qkv_name"] = name
            # Find bias
            bias_name = name.replace("weight", "bias")
            for n2, p2 in layer.named_parameters():
                if n2 == bias_name:
                    result["qkv_bias"] = p2
                    break
            result["fused_qkv"] = True
            break

    # Try separate Q/K/V (BERT style or decoder style)
    if "qkv_weight" not in result:
        q_patterns = ["query.weight", "q_proj.weight", "self.query.weight"]
        k_patterns = ["key.weight", "k_proj.weight", "self.key.weight"]
        v_patterns = ["value.weight", "v_proj.weight", "self.value.weight"]

        for name, param in layer.named_parameters():
            for pat in q_patterns:
                if name.endswith(pat):
                    result["q_weight"] = param
                    bias_name = name.replace("weight", "bias")
                    for n2, p2 in layer.named_parameters():
                        if n2 == bias_name:
                            result["q_bias"] = p2
            for pat in k_patterns:
                if name.endswith(pat):
                    result["k_weight"] = param
                    bias_name = name.replace("weight", "bias")
                    for n2, p2 in layer.named_parameters():
                        if n2 == bias_name:
                            result["k_bias"] = p2
            for pat in v_patterns:
                if name.endswith(pat):
                    result["v_weight"] = param
                    bias_name = name.replace("weight", "bias")
                    for n2, p2 in layer.named_parameters():
                        if n2 == bias_name:
                            result["v_bias"] = p2

        if "q_weight" in result:
            result["fused_qkv"] = False

    # Find output projection
    o_patterns = ["o_proj.weight", "output.dense.weight", "attention.output.dense.weight",
                  "out_proj.weight", "dense.weight"]
    for name, param in layer.named_parameters():
        for pat in o_patterns:
            if name.endswith(pat) and "attention" in name.lower():
                result["o_weight"] = param
                bias_name = name.replace("weight", "bias")
                for n2, p2 in layer.named_parameters():
                    if n2 == bias_name:
                        result["o_bias"] = p2
                break
        if "o_weight" in result:
            break

    # Fallback: find o_proj by shape matching
    if "o_weight" not in result:
        for name, param in layer.named_parameters():
            if "o_proj" in name and "weight" in name:
                result["o_weight"] = param
                break

    if "o_weight" not in result:
        return None
    if "qkv_weight" not in result and "q_weight" not in result:
        return None

    return result


def _score_heads(o_weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Score attention heads by L1 norm of their output projection weights."""
    w = o_weight.data.float()
    # o_proj: [hidden, num_heads*head_dim] — each head occupies head_dim columns
    if w.shape[1] == num_heads * head_dim:
        # [hidden, n_heads*head_dim] → group by head
        w_heads = w.view(w.shape[0], num_heads, head_dim)  # [H, n, d]
        scores = w_heads.abs().sum(dim=(0, 2))  # [n]
    elif w.shape[0] == num_heads * head_dim:
        # Transposed: [n_heads*head_dim, hidden]
        w_heads = w.view(num_heads, head_dim, w.shape[1])
        scores = w_heads.abs().sum(dim=(1, 2))
    else:
        # Fallback: uniform scores
        scores = torch.ones(num_heads)
    return scores


def _apply_head_pruning(attn_info: dict, keep_indices: torch.Tensor, num_heads: int, head_dim: int):
    """In-place prune attention weights to keep only selected heads."""
    keep = keep_indices.tolist()
    n_keep = len(keep)

    # Build index mask for head_dim-sized chunks
    keep_dims = []
    for h in keep:
        keep_dims.extend(range(h * head_dim, (h + 1) * head_dim))
    keep_dims = torch.tensor(keep_dims, dtype=torch.long)

    if attn_info.get("fused_qkv"):
        # QKV fused: [3*n_heads*head_dim, hidden]
        w = attn_info["qkv_weight"]
        # Split into Q, K, V chunks
        qkv_dim = num_heads * head_dim
        q_w = w.data[:qkv_dim]
        k_w = w.data[qkv_dim:2*qkv_dim]
        v_w = w.data[2*qkv_dim:3*qkv_dim]

        new_q = q_w[keep_dims]
        new_k = k_w[keep_dims]
        new_v = v_w[keep_dims]
        new_w = torch.cat([new_q, new_k, new_v], dim=0)

        # Replace parameter data in-place via module
        _replace_param_data(attn_info["qkv_weight"], new_w)

        if "qkv_bias" in attn_info:
            b = attn_info["qkv_bias"]
            q_b, k_b, v_b = b.data[:qkv_dim], b.data[qkv_dim:2*qkv_dim], b.data[2*qkv_dim:]
            new_b = torch.cat([q_b[keep_dims], k_b[keep_dims], v_b[keep_dims]])
            _replace_param_data(b, new_b)
    else:
        # Separate Q, K, V
        for key in ["q_weight", "k_weight", "v_weight"]:
            if key in attn_info:
                w = attn_info[key]
                if w.shape[0] == num_heads * head_dim:
                    _replace_param_data(w, w.data[keep_dims])
                elif w.shape[1] == num_heads * head_dim:
                    _replace_param_data(w, w.data[:, keep_dims])

        for key in ["q_bias", "k_bias", "v_bias"]:
            if key in attn_info:
                b = attn_info[key]
                if b.shape[0] == num_heads * head_dim:
                    _replace_param_data(b, b.data[keep_dims])

    # Output projection
    o_w = attn_info["o_weight"]
    if o_w.shape[1] == num_heads * head_dim:
        _replace_param_data(o_w, o_w.data[:, keep_dims])
    elif o_w.shape[0] == num_heads * head_dim:
        _replace_param_data(o_w, o_w.data[keep_dims])

    if "o_bias" in attn_info:
        # o_bias is [hidden_dim], doesn't change with head pruning
        pass


def _replace_param_data(param: nn.Parameter, new_data: torch.Tensor):
    """Replace parameter data in-place (keeps the same Parameter object)."""
    param.data = new_data


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

    # Fix config for head-pruned models:
    # If attention weights don't match config expectations, adjust config
    config_path_tmp = os.path.join(hf_tmp, "config.json")
    if os.path.exists(config_path_tmp):
        with open(config_path_tmp, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Detect actual attention dim from saved weights
        import safetensors.torch
        for wf in ["model.safetensors", "pytorch_model.bin"]:
            wp = os.path.join(hf_tmp, wf)
            if os.path.exists(wp):
                if wf.endswith(".safetensors"):
                    sd = safetensors.torch.load_file(wp)
                else:
                    sd = torch.load(wp, map_location="cpu", weights_only=True)
                # Find a qkv or query weight to detect actual head count
                for k, v in sd.items():
                    if "qkv_proj.weight" in k:
                        # fused QKV: [3*attn_dim, hidden]
                        attn_dim = v.shape[0] // 3
                        n_heads_cfg = cfg.get("num_attention_heads", 12)
                        if "head_dim" in cfg:
                            hd = cfg["head_dim"]
                        else:
                            hd = cfg["hidden_size"] // n_heads_cfg
                        actual_heads = attn_dim // hd
                        cfg["num_attention_heads"] = actual_heads
                        cfg["head_dim"] = hd
                        break
                    if "query.weight" in k or "q_proj.weight" in k:
                        attn_dim = v.shape[0]
                        n_heads_cfg = cfg.get("num_attention_heads", 12)
                        if "head_dim" in cfg:
                            hd = cfg["head_dim"]
                        else:
                            hd = cfg["hidden_size"] // n_heads_cfg
                        actual_heads = attn_dim // hd
                        cfg["num_attention_heads"] = actual_heads
                        cfg["head_dim"] = hd
                        break
                break
        with open(config_path_tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

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

    # Patch custom model code to support head_dim from config (for head-pruned models)
    _patch_modeling_for_head_pruning(hf_tmp)

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


def _patch_modeling_for_head_pruning(model_dir: str):
    """Patch custom modeling.py to support head_dim from config.

    After head pruning, num_attention_heads changes but head_dim should stay
    the same (e.g. 64). Custom model code often computes head_dim as
    hidden_size // num_attention_heads, which breaks. This patches the code
    to use config.head_dim when available.
    """
    modeling_path = os.path.join(model_dir, "modeling.py")
    if not os.path.exists(modeling_path):
        return

    with open(modeling_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Check if there's a head_dim in config.json
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "head_dim" not in cfg:
        return

    # Patch: replace `int(config.hidden_size / config.num_attention_heads)`
    # with `getattr(config, 'head_dim', int(config.hidden_size / config.num_attention_heads))`
    old_pattern = "int(config.hidden_size / config.num_attention_heads)"
    new_pattern = "getattr(config, 'head_dim', int(config.hidden_size / config.num_attention_heads))"

    if old_pattern in code and new_pattern not in code:
        code = code.replace(old_pattern, new_pattern)

        # Also patch o_proj to use all_head_size instead of hidden_size
        # Original: nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # For the output projection after head pruning
        old_o = "self.o_proj = nn.Linear(config.hidden_size, config.hidden_size"
        new_o = "self.o_proj = nn.Linear(self.all_head_size, config.hidden_size"
        if old_o in code:
            code = code.replace(old_o, new_o)

        with open(modeling_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  Patched {modeling_path} for head_dim support")


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
