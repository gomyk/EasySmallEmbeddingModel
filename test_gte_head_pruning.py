"""Test GTE compression: layer pruning + head pruning + vocab pruning, then 20-epoch distill."""

import os
import torch
from smallmodel.arch import (
    create_pruned_student,
    prune_attention_heads,
    collect_corpus_tokens,
    prune_tokenizer_and_embeddings,
    save_as_sentence_transformer,
)
from smallmodel.data import load_distill_corpus
from smallmodel.distill import distill
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import shutil

OUTPUT_DIR = "output/gte_head_test"
TEACHER_ID = "alibaba-NLP/gte-multilingual-base"

# Config: L3, 4 heads (from 12), vocab pruned by corpus
# L3 H4 with 3072 FFN ≈ 60-70MB depending on vocab
LAYER_INDICES = [0, 6, 11]  # 3 layers, uniform
NUM_HEADS = 4  # from 12
DISTILL_EPOCHS = 20

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    student_path = os.path.join(OUTPUT_DIR, "gte_L3_H4")

    print("=" * 60)
    print("GTE Head Pruning Test")
    print(f"  Layers: {LAYER_INDICES} (3 from 12)")
    print(f"  Heads: {NUM_HEADS} (from 12, head_dim=64)")
    print(f"  Vocab: corpus-based pruning")
    print(f"  Distill: {DISTILL_EPOCHS} epochs")
    print("=" * 60)

    # Step 1: Layer pruning
    print("\n[1/5] Layer pruning...")
    student, tokenizer = create_pruned_student(
        TEACHER_ID, LAYER_INDICES,
        layer_accessor="encoder.layer",
        trust_remote_code=True,
    )
    print(f"  Layers: 12 -> {len(LAYER_INDICES)}")

    # Step 2: Head pruning
    print("\n[2/5] Attention head pruning...")
    student = prune_attention_heads(student, NUM_HEADS, layer_accessor="encoder.layer")

    # Check actual param count
    total_params = sum(p.numel() for p in student.parameters())
    print(f"  After head pruning: {total_params:,} params ({total_params*4/1024**2:.1f}MB FP32)")

    # Step 3: Vocab pruning
    print("\n[3/5] Loading corpus for vocab pruning...")
    corpus = load_distill_corpus(max_per_lang=5000, cache_dir=os.path.join(OUTPUT_DIR, "data"))
    keep_ids = collect_corpus_tokens(tokenizer, texts=corpus, vocab_keep_ratio=0.95)
    print(f"  Corpus tokens to keep: {len(keep_ids):,} / 250,048")

    hf_tmp = os.path.join(student_path, "_hf_pruned")
    student = prune_tokenizer_and_embeddings(student, tokenizer, keep_ids, hf_tmp)
    tokenizer = AutoTokenizer.from_pretrained(hf_tmp, trust_remote_code=True)
    print(f"  Vocab: 250,048 -> {student.config.vocab_size:,}")

    # Step 4: Save
    print("\n[4/5] Saving as SentenceTransformer...")
    save_as_sentence_transformer(student, tokenizer, student_path)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    # Verify
    st = SentenceTransformer(student_path, trust_remote_code=True)
    emb = st.encode(["Hello world", "안녕하세요", "こんにちは"])
    print(f"  Output shape: {emb.shape}")
    total_params = sum(p.numel() for p in st.parameters())
    print(f"  Total params: {total_params:,} ({total_params*4/1024**2:.1f}MB FP32)")

    # Check actual file size
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        for base in [student_path, os.path.join(student_path, "0_Transformer")]:
            fp = os.path.join(base, fname)
            if os.path.exists(fp):
                size_mb = os.path.getsize(fp) / (1024**2)
                print(f"  File size: {size_mb:.1f}MB ({fname})")
    del st

    # Step 5: Distillation
    print(f"\n[5/5] Distillation ({DISTILL_EPOCHS} epochs)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    distilled_path = distill(
        teacher_name=TEACHER_ID,
        student_path=student_path,
        epochs=DISTILL_EPOCHS,
        batch_size=32,
        lr=2e-5,
        patience=5,
        device=device,
        trust_remote_code=True,
    )

    print(f"\nDone! Distilled model at: {distilled_path}")

    # Final check
    st = SentenceTransformer(distilled_path, trust_remote_code=True)
    emb = st.encode(["Hello world", "안녕하세요", "こんにちは"])
    print(f"  Final output shape: {emb.shape}")


if __name__ == "__main__":
    main()
