"""Model size estimation and optimal config search."""

from __future__ import annotations

from smallmodel.teachers import TEACHERS


def make_uniform_indices(num_layers: int, target_count: int) -> list[int]:
    """Generate evenly-spaced layer indices."""
    return [round(i * (num_layers - 1) / (target_count - 1)) for i in range(target_count)]


def estimate_size(
    layer_indices: list[int],
    hidden_dim: int = 384,
    vocab_size: int = 40000,
    intermediate_size: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    has_glu: bool = False,
    is_decoder: bool = False,
) -> dict:
    """Estimate FP32 model size in MB."""
    if intermediate_size is None:
        intermediate_size = hidden_dim * 4

    embed_params = vocab_size * hidden_dim
    if is_decoder:
        embed_params += hidden_dim
    else:
        embed_params += hidden_dim + 514 * hidden_dim + 2 * hidden_dim

    if num_attention_heads and num_kv_heads:
        hd = head_dim if head_dim else (hidden_dim // num_attention_heads)
        q_dim = num_attention_heads * hd
        kv_dim = num_kv_heads * hd
        attn_params = hidden_dim * q_dim + hidden_dim * kv_dim * 2 + q_dim * hidden_dim
    else:
        attn_params = 4 * hidden_dim * hidden_dim

    ffn_multiplier = 3 if has_glu else 2
    ffn_params = ffn_multiplier * hidden_dim * intermediate_size
    layer_params = attn_params + ffn_params + 4 * hidden_dim
    total_params = embed_params + len(layer_indices) * layer_params
    fp32_mb = total_params * 4 / (1024 ** 2)

    return {"fp32_mb": round(fp32_mb, 1), "total_params": total_params}


def estimate_for_teacher(
    teacher_key: str,
    layer_indices: list[int],
    vocab_size: int | None = None,
    hidden_dim: int | None = None,
    intermediate_size: int | None = None,
) -> dict:
    """Estimate size using teacher config defaults."""
    t = TEACHERS[teacher_key]
    h = hidden_dim if hidden_dim is not None else t["hidden_dim"]
    inter = intermediate_size if intermediate_size is not None else t["intermediate_size"]
    v = vocab_size if vocab_size is not None else t["vocab_size"]

    n_heads = t.get("num_attention_heads")
    n_kv = t.get("num_kv_heads")
    hd = t.get("head_dim")

    if h != t["hidden_dim"] and n_heads:
        ratio = h / t["hidden_dim"]
        n_heads = max(1, int(n_heads * ratio))
        while h % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        if n_kv:
            n_kv = max(1, int(n_kv * ratio))
            while n_kv > 1 and (n_heads % n_kv != 0 or h % n_kv != 0):
                n_kv -= 1
        if hd:
            hd = h // n_heads

    return estimate_size(
        layer_indices, h, v, inter,
        num_attention_heads=n_heads, num_kv_heads=n_kv, head_dim=hd,
        has_glu=t.get("has_glu", False), is_decoder=t.get("is_decoder", False),
    )


def find_optimal_config(
    teacher_key: str,
    max_params: int = 20_000_000,
    max_fp32_mb: float = 50.0,
    min_layers: int = 4,
    estimated_vocab_size: int | None = None,
    corpus_vocab_size: int | None = None,
) -> dict:
    """Search for the optimal model config within size constraints.

    Priority: hidden dim preservation > layer count > vocab size.
    """
    t = TEACHERS[teacher_key]
    param_limit_from_mb = int(max_fp32_mb * 1024 * 1024 / 4)
    effective_max = min(max_params, param_limit_from_mb)

    hidden_dim = t["hidden_dim"]
    inter_size = t["intermediate_size"]
    vocab_cap = estimated_vocab_size or corpus_vocab_size or t["vocab_size"]
    vocab_floor = estimated_vocab_size if estimated_vocab_size else 3000

    def _calc_vocab_budget(layer_indices, h, inter):
        s_zero = estimate_for_teacher(teacher_key, layer_indices, 0,
                                      hidden_dim=h, intermediate_size=inter)
        remaining = effective_max - s_zero["total_params"]
        max_v = int(remaining / h) if h > 0 else 0
        return min(max(max_v, 0), vocab_cap)

    indices = make_uniform_indices(t["num_layers"], min_layers)

    v_budget = _calc_vocab_budget(indices, hidden_dim, inter_size)
    if v_budget >= vocab_floor:
        return {
            "layer_indices": indices,
            "hidden_dim": hidden_dim,
            "intermediate_size": inter_size,
            "target_vocab": v_budget,
            "needs_hidden_reduction": False,
        }

    lo, hi = 64, hidden_dim - 64
    best_h = 64
    while lo <= hi:
        mid = ((lo + hi) // 2 // 64) * 64
        if mid < 64:
            mid = 64
        ratio = mid / hidden_dim
        scaled_inter = max(64, (int(inter_size * ratio) // 64) * 64)
        v_budget = _calc_vocab_budget(indices, mid, scaled_inter)
        if v_budget >= vocab_floor:
            best_h = mid
            lo = mid + 64
        else:
            hi = mid - 64

    best_ratio = best_h / hidden_dim
    best_inter = max(64, (int(inter_size * best_ratio) // 64) * 64)
    v_budget = _calc_vocab_budget(indices, best_h, best_inter)

    return {
        "layer_indices": indices,
        "hidden_dim": best_h,
        "intermediate_size": best_inter,
        "target_vocab": max(v_budget, vocab_floor),
        "needs_hidden_reduction": best_h < hidden_dim,
    }
