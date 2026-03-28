"""Background distillation runner with progress streaming."""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DistillProgress:
    """Thread-safe progress emitter for SSE streaming."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.running = False
        self.finished = False
        self.error: str | None = None

    def emit(self, event: str, data: dict):
        self.q.put({"event": event, "data": data})

    def stream(self):
        """Generator for SSE events."""
        while not self.finished or not self.q.empty():
            try:
                msg = self.q.get(timeout=1.0)
                yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
            except queue.Empty:
                yield f"event: ping\ndata: {{}}\n\n"
        # Final done event
        if self.error:
            yield f"event: error\ndata: {json.dumps({'message': self.error})}\n\n"
        yield f"event: done\ndata: {json.dumps({'finished': True})}\n\n"


class _TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def run_distillation(
    teacher_path: str,
    student_path: str,
    output_path: str,
    texts: list[str],
    progress: DistillProgress,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 64,
    cos_weight: float = 0.5,
    mse_weight: float = 1.0,
    patience: int = 3,
    device: str = "cpu",
    trust_remote_code_teacher: bool = False,
    save_every_epoch: bool = True,
):
    """Run distillation in a background thread with progress reporting."""
    from sentence_transformers import SentenceTransformer

    progress.running = True

    try:
        # ── Setup ──
        progress.emit("status", {"message": f"Loading teacher: {teacher_path}...", "phase": "setup"})
        teacher = SentenceTransformer(teacher_path, device=device,
                                      trust_remote_code=trust_remote_code_teacher)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        progress.emit("status", {"message": f"Loading student: {student_path}...", "phase": "setup"})
        student = SentenceTransformer(student_path, device=device, trust_remote_code=True)
        student.train()

        teacher_dim = teacher.get_sentence_embedding_dimension()
        student_dim = student.get_sentence_embedding_dimension()

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
        dataset = _TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        loss_fn = nn.MSELoss()
        tokenizer = student.tokenizer
        student_transformer = student[0].auto_model

        proj = None
        if student_dim != teacher_dim:
            proj = nn.Linear(student_dim, teacher_dim).to(device)
            optimizer = torch.optim.AdamW(
                list(student.parameters()) + list(proj.parameters()),
                lr=lr, weight_decay=0.01,
            )
            progress.emit("status", {
                "message": f"Projection layer added: {student_dim}d -> {teacher_dim}d",
                "phase": "setup",
            })

        total_batches = len(dataloader)
        best_loss = float("inf")
        no_improve_count = 0
        epoch_times = []
        os.makedirs(output_path, exist_ok=True)

        progress.emit("config", {
            "teacher": teacher_path,
            "student": student_path,
            "teacher_dim": teacher_dim,
            "student_dim": student_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "total_batches": total_batches,
            "total_texts": len(texts),
            "device": device,
            "patience": patience,
        })

        # ── Training Loop ──
        global_step = 0
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            n_batches = 0
            student_transformer.train()
            if proj:
                proj.train()

            progress.emit("epoch_start", {
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "best_loss": round(best_loss, 6) if best_loss < float("inf") else None,
            })

            batch_losses = []

            for batch_idx, batch_texts in enumerate(dataloader):
                batch_start = time.time()
                try:
                    with torch.no_grad():
                        teacher_emb = teacher.encode(
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
                student_emb = torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

                projected = proj(student_emb) if proj else student_emb

                mse_loss = loss_fn(projected, teacher_emb)
                cos_loss = 1 - F.cosine_similarity(projected, teacher_emb).mean()
                total_loss = mse_weight * mse_loss + cos_weight * cos_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = total_loss.item()
                epoch_loss += loss_val
                n_batches += 1
                global_step += 1
                batch_losses.append(loss_val)

                batch_time = time.time() - batch_start

                # Estimate remaining time
                elapsed_epoch = time.time() - epoch_start
                if batch_idx > 0:
                    avg_batch_time = elapsed_epoch / (batch_idx + 1)
                    remaining_batches = total_batches - batch_idx - 1
                    remaining_epochs = epochs - epoch - 1
                    eta_epoch = avg_batch_time * remaining_batches
                    eta_total = eta_epoch + remaining_epochs * (elapsed_epoch / (batch_idx + 1) * total_batches)
                else:
                    eta_epoch = 0
                    eta_total = 0

                # Emit every 5 batches or last batch
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    progress.emit("batch", {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "loss": round(loss_val, 6),
                        "mse_loss": round(mse_loss.item(), 6),
                        "cos_loss": round(cos_loss.item(), 6),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                        "batch_time_ms": round(batch_time * 1000, 1),
                        "eta_epoch_sec": round(eta_epoch, 1),
                        "eta_total_sec": round(eta_total, 1),
                        "global_step": global_step,
                        "total_steps": total_steps,
                    })

            # ── Epoch End ──
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_loss = epoch_loss / max(n_batches, 1)

            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Estimate remaining
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - epoch - 1
            eta_remaining = avg_epoch_time * remaining_epochs

            progress.emit("epoch_end", {
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "avg_loss": round(avg_loss, 6),
                "best_loss": round(best_loss, 6),
                "improved": improved,
                "no_improve_count": no_improve_count,
                "patience": patience,
                "epoch_time_sec": round(epoch_time, 1),
                "eta_remaining_sec": round(eta_remaining, 1),
                "losses": [round(l, 6) for l in batch_losses[::max(1, len(batch_losses)//20)]],
            })

            # Save model
            if save_every_epoch or improved:
                save_label = "best" if improved else f"epoch_{epoch+1}"
                save_dir = output_path if improved else os.path.join(output_path, f"epoch_{epoch+1}")

                _save_student(student_transformer, tokenizer, student_dim, save_dir)

                progress.emit("model_saved", {
                    "epoch": epoch + 1,
                    "path": save_dir,
                    "label": save_label,
                    "loss": round(avg_loss, 6),
                })

            # Early stopping
            if no_improve_count >= patience:
                progress.emit("early_stop", {
                    "epoch": epoch + 1,
                    "best_loss": round(best_loss, 6),
                    "patience": patience,
                })
                break

        # ── Finish ──
        total_time = sum(epoch_times)
        progress.emit("complete", {
            "best_loss": round(best_loss, 6),
            "total_epochs_run": epoch + 1,
            "total_time_sec": round(total_time, 1),
            "output_path": output_path,
        })

    except Exception as e:
        import traceback
        progress.error = str(e)
        progress.emit("error", {"message": str(e), "traceback": traceback.format_exc()})

    finally:
        progress.running = False
        progress.finished = True


def _save_student(student_transformer, tokenizer, student_dim, save_dir):
    """Save student as SentenceTransformer format."""
    from sentence_transformers import SentenceTransformer, models as st_models

    os.makedirs(save_dir, exist_ok=True)
    student_transformer.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    word_model = st_models.Transformer(
        save_dir,
        config_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )
    pool_model = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    st_model = SentenceTransformer(modules=[word_model, pool_model])
    st_model.save(save_dir)
