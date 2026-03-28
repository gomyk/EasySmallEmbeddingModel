"""Knowledge distillation: teacher embedding -> student embedding alignment."""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from smallmodel.teachers import TEACHERS
from smallmodel.data import load_mteb_task_texts


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def distill(
    teacher_name: str,
    student_path: str,
    texts: list[str] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 64,
    device: str | None = None,
    cos_weight: float = 0.5,
    mse_weight: float = 1.0,
    suffix: str = "_distilled",
    trust_remote_code: bool = False,
    patience: int = 3,
) -> str:
    """Distill teacher embeddings into a student model.

    Returns the path to the distilled model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if texts is None:
        texts = load_mteb_task_texts()

    print(f"\nDistilling: {os.path.basename(student_path)}")
    print(f"  Teacher: {teacher_name}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Device: {device}")

    teacher = SentenceTransformer(teacher_name, device=device,
                                  trust_remote_code=trust_remote_code)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = SentenceTransformer(student_path, device=device, trust_remote_code=True)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_loss = float("inf")
    no_improve_count = 0
    loss_fn = nn.MSELoss()
    tokenizer = student.tokenizer
    student_transformer = student[0].auto_model

    teacher_dim = teacher.get_sentence_embedding_dimension()
    student_dim = student.get_sentence_embedding_dimension()
    proj = None
    if student_dim != teacher_dim:
        proj = nn.Linear(student_dim, teacher_dim).to(device)
        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(proj.parameters()),
            lr=lr, weight_decay=0.01,
        )

    distilled_path = student_path + suffix

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        student_transformer.train()
        if proj:
            proj.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_texts in pbar:
            try:
                with torch.no_grad():
                    teacher_embeddings = teacher.encode(
                        list(batch_texts), convert_to_tensor=True,
                        show_progress_bar=False, device=device,
                    ).clone()

                encoded = tokenizer(
                    list(batch_texts), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
            except Exception:
                continue

            model_output = student_transformer(**encoded)
            token_emb = model_output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            student_embeddings = torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

            projected = proj(student_embeddings) if proj else student_embeddings

            loss = loss_fn(projected, teacher_embeddings)
            cos_loss = 1 - F.cosine_similarity(projected, teacher_embeddings).mean()
            total_loss = mse_weight * loss + cos_weight * cos_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            os.makedirs(distilled_path, exist_ok=True)
            student_transformer.save_pretrained(distilled_path)
            tokenizer.save_pretrained(distilled_path)
            from sentence_transformers import models as st_models
            word_model = st_models.Transformer(
                distilled_path,
                config_args={"trust_remote_code": True},
                model_args={"trust_remote_code": True},
                tokenizer_args={"trust_remote_code": True},
            )
            pool_model = st_models.Pooling(word_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True)
            st_model = SentenceTransformer(modules=[word_model, pool_model])
            st_model.save(distilled_path)
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"Distillation complete. Best loss: {best_loss:.4f}")
    return distilled_path


def distill_two_stage(
    teacher_key: str,
    student_name: str,
    students_dir: str,
    texts: list[str] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str | None = None,
    patience: int = 3,
) -> str:
    """Two-stage distillation: Teacher -> Intermediate -> Student."""
    t = TEACHERS[teacher_key]
    teacher_name = t["model_id"]

    intermediate_path = os.path.join(students_dir, f"{teacher_key}_intermediate")
    if not os.path.exists(intermediate_path):
        raise FileNotFoundError(f"Intermediate model not found: {intermediate_path}")

    student_path = os.path.join(students_dir, student_name)
    if not os.path.exists(student_path):
        raise FileNotFoundError(f"Student model not found: {student_path}")

    if texts is None:
        texts = load_mteb_task_texts()

    print(f"\n2-STAGE DISTILLATION")
    print(f"  Teacher: {teacher_name}")
    print(f"  Intermediate: {teacher_key}_intermediate")
    print(f"  Student: {student_name}")

    # Stage 1: Teacher -> Intermediate
    print(f"\n  STAGE 1/2: Teacher -> Intermediate")
    distill(
        teacher_name=teacher_name, student_path=intermediate_path,
        texts=texts, epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, trust_remote_code=t["trust_remote_code"], patience=patience,
    )

    # Stage 2: Intermediate_distilled -> Student
    intermediate_distilled = intermediate_path + "_distilled"
    if not os.path.exists(intermediate_distilled):
        raise FileNotFoundError(f"Stage 1 output not found: {intermediate_distilled}")

    print(f"\n  STAGE 2/2: Intermediate -> Student")
    result = distill(
        teacher_name=intermediate_distilled, student_path=student_path,
        texts=texts, epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, trust_remote_code=True, patience=patience,
    )

    print(f"\n2-Stage distillation complete!")
    return result
