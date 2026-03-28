"""SmallModel - High-level API for model compression.

Example::

    from smallmodel import SmallModel

    # Quick compress
    sm = SmallModel.from_teacher("gte")
    sm.compress(max_params=20_000_000, max_fp32_mb=50.0)
    sm.distill(epochs=10)
    sm.evaluate()

    # Custom layer selection
    sm = SmallModel.from_teacher("gte", layer_indices=[0, 3, 6, 11])
    sm.create()

    # Interactive web editor
    sm = SmallModel.from_teacher("gte")
    sm.serve()  # Opens localhost web UI
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from smallmodel.teachers import TEACHERS, get_teacher
from smallmodel.sizing import (
    make_uniform_indices, estimate_size, estimate_for_teacher,
    find_optimal_config,
)


class SmallModel:
    """High-level API for creating compressed embedding models."""

    def __init__(
        self,
        teacher_key: str,
        layer_indices: list[int] | None = None,
        hidden_dim: int | None = None,
        intermediate_size: int | None = None,
        vocab_keep_ratio: float | None = None,
        max_vocab: int | None = None,
        output_dir: str = "output",
    ):
        self.teacher_key = teacher_key
        self.teacher = get_teacher(teacher_key)
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim or self.teacher["hidden_dim"]
        self.intermediate_size = intermediate_size or self.teacher["intermediate_size"]
        self.vocab_keep_ratio = vocab_keep_ratio
        self.max_vocab = max_vocab
        self.output_dir = output_dir
        self._student_path: str | None = None
        self._distilled_path: str | None = None
        self._intermediate_path: str | None = None
        self._corpus_texts: list[str] | None = None
        self._needs_hidden_reduction = False
        self._needs_two_stage = False

    @classmethod
    def from_teacher(cls, teacher_key: str, **kwargs) -> SmallModel:
        """Create a SmallModel from a registered teacher.

        Args:
            teacher_key: Key in TEACHERS registry (e.g. "gte", "modernbert").
            **kwargs: Passed to SmallModel.__init__.
        """
        return cls(teacher_key, **kwargs)

    # ── Size Estimation ──────────────────────────────────────────

    def estimate(self, layer_indices: list[int] | None = None) -> dict:
        """Estimate model size for given (or current) layer config."""
        indices = layer_indices or self.layer_indices
        if indices is None:
            indices = list(range(self.teacher["num_layers"]))
        return estimate_for_teacher(
            self.teacher_key, indices,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size,
        )

    def get_teacher_info(self) -> dict[str, Any]:
        """Return teacher model metadata for display."""
        t = self.teacher
        teacher_layers = list(range(t["num_layers"]))
        teacher_est = estimate_for_teacher(self.teacher_key, teacher_layers)
        return {
            "key": self.teacher_key,
            "model_id": t["model_id"],
            "short_name": t["short_name"],
            "num_layers": t["num_layers"],
            "hidden_dim": t["hidden_dim"],
            "intermediate_size": t["intermediate_size"],
            "vocab_size": t["vocab_size"],
            "layer_accessor": t["layer_accessor"],
            "tokenizer_type": t.get("tokenizer_type", "unknown"),
            "is_decoder": t.get("is_decoder", False),
            "has_glu": t.get("has_glu", False),
            "total_params": teacher_est["total_params"],
            "fp32_mb": teacher_est["fp32_mb"],
        }

    # ── Corpus Loading ───────────────────────────────────────────

    def load_corpus(self, max_per_lang: int = 5000) -> list[str]:
        """Load distillation corpus (cached)."""
        if self._corpus_texts is None:
            from smallmodel.data import load_distill_corpus
            self._corpus_texts = load_distill_corpus(
                max_per_lang=max_per_lang,
                cache_dir=os.path.join(self.output_dir, "data", "distill_corpus"),
            )
        return self._corpus_texts

    # ── Student Creation ─────────────────────────────────────────

    def create(self, name: str | None = None, no_prune: bool = False) -> str:
        """Create a student via layer pruning + vocab pruning.

        Uses self.layer_indices (must be set). Returns the student path.
        """
        from smallmodel.arch import (
            create_pruned_student, collect_corpus_tokens,
            prune_tokenizer_and_embeddings, save_as_sentence_transformer,
        )
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer

        if self.layer_indices is None:
            raise ValueError("layer_indices must be set. Use .compress() or set manually.")

        t = self.teacher
        name = name or f"{self.teacher_key}_L{len(self.layer_indices)}"
        save_path = os.path.join(self.output_dir, "students", self.teacher_key, name)
        os.makedirs(save_path, exist_ok=True)

        est = self.estimate()
        print(f"\nCreating {name}: layers={self.layer_indices}")
        print(f"  Teacher: {t['model_id']}")
        print(f"  Estimated: {est['fp32_mb']}MB ({est['total_params']:,} params)")

        student_hf, tokenizer = create_pruned_student(
            t["model_id"], self.layer_indices,
            layer_accessor=t["layer_accessor"],
            trust_remote_code=t["trust_remote_code"],
        )
        print(f"  Layer pruning: {t['num_layers']} -> {len(self.layer_indices)} layers")

        if not no_prune:
            corpus_texts = self.load_corpus()
            keep_ids = collect_corpus_tokens(
                tokenizer, texts=corpus_texts,
                max_vocab=self.max_vocab,
                vocab_keep_ratio=self.vocab_keep_ratio,
            )
            hf_tmp = os.path.join(save_path, "_hf_pruned")
            student_hf = prune_tokenizer_and_embeddings(student_hf, tokenizer, keep_ids, hf_tmp)
            print(f"  Vocab pruned: {t['vocab_size']:,} -> {student_hf.config.vocab_size:,}")
            tokenizer = AutoTokenizer.from_pretrained(hf_tmp, trust_remote_code=t["trust_remote_code"])
            save_as_sentence_transformer(student_hf, tokenizer, save_path)
            shutil.rmtree(hf_tmp, ignore_errors=True)
        else:
            save_as_sentence_transformer(student_hf, tokenizer, save_path)

        # Sanity check
        try:
            st = SentenceTransformer(save_path, trust_remote_code=True)
            embeddings = st.encode(["Hello world", "안녕하세요"])
            print(f"  Sanity check: output shape = {embeddings.shape}")
            del st
        except Exception as e:
            print(f"  Sanity check failed: {e}")

        self._student_path = save_path
        print(f"  Saved to {save_path}")
        return save_path

    # ── Compress (auto-optimal) ──────────────────────────────────

    def compress(
        self,
        max_params: int = 20_000_000,
        max_fp32_mb: float = 50.0,
        min_layers: int = 4,
        vocab_percentile: float = 0.95,
        min_vocab: int | None = None,
        use_pca: bool = False,
    ) -> str:
        """Create an optimally compressed model within size constraints.

        Automatically determines layer count, hidden dim, and vocab size.
        If compression ratio > 10x, also creates an intermediate model for
        two-stage distillation.

        Returns the final student path.
        """
        from smallmodel.arch import (
            create_pruned_student, collect_corpus_tokens,
            prune_tokenizer_and_embeddings, save_as_sentence_transformer,
            reduce_hidden_dim, reduce_hidden_dim_pca,
        )
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer

        t = self.teacher
        print(f"\nCompressing: {t['model_id']}")
        print(f"  Constraints: max {max_params/1e6:.0f}M params, {max_fp32_mb}MB FP32")

        # Phase 1: Vocab analysis
        corpus_texts = self.load_corpus()
        tokenizer = AutoTokenizer.from_pretrained(t["model_id"], trust_remote_code=t["trust_remote_code"])
        keep_ids_all = collect_corpus_tokens(tokenizer, texts=corpus_texts, vocab_keep_ratio=vocab_percentile)
        corpus_vocab_size = len(keep_ids_all)

        if min_vocab and corpus_vocab_size < min_vocab:
            keep_ids_all = collect_corpus_tokens(tokenizer, texts=corpus_texts, max_vocab=min_vocab)
            corpus_vocab_size = len(keep_ids_all)

        print(f"  Corpus vocab: {t['vocab_size']:,} -> {corpus_vocab_size:,}")

        # Phase 2: Joint optimization
        opt = find_optimal_config(self.teacher_key, max_params, max_fp32_mb, min_layers,
                                  corpus_vocab_size=corpus_vocab_size)

        self.layer_indices = opt["layer_indices"]
        self.hidden_dim = opt["hidden_dim"]
        self.intermediate_size = opt["intermediate_size"]
        self._needs_hidden_reduction = opt["needs_hidden_reduction"]

        target_vocab = opt["target_vocab"]
        if target_vocab < corpus_vocab_size:
            keep_ids = collect_corpus_tokens(tokenizer, texts=corpus_texts, max_vocab=target_vocab)
        else:
            keep_ids = keep_ids_all
        actual_vocab = len(keep_ids)

        if actual_vocab > target_vocab:
            reopt = find_optimal_config(self.teacher_key, max_params, max_fp32_mb, min_layers,
                                        estimated_vocab_size=actual_vocab)
            self.layer_indices = reopt["layer_indices"]
            self.hidden_dim = reopt["hidden_dim"]
            self.intermediate_size = reopt["intermediate_size"]

        est = estimate_for_teacher(self.teacher_key, self.layer_indices, actual_vocab,
                                   hidden_dim=self.hidden_dim, intermediate_size=self.intermediate_size)
        print(f"  Config: {len(self.layer_indices)}L / {self.hidden_dim}d / {actual_vocab:,} vocab")
        print(f"  Estimated: {est['total_params']:,} params, {est['fp32_mb']}MB")

        # Check if two-stage is needed
        teacher_all = list(range(t["num_layers"]))
        teacher_est = estimate_for_teacher(self.teacher_key, teacher_all, t["vocab_size"])
        ratio = teacher_est["total_params"] / max(est["total_params"], 1)
        self._needs_two_stage = ratio > 10

        students_dir = os.path.join(self.output_dir, "students", self.teacher_key)
        os.makedirs(students_dir, exist_ok=True)

        def _build_model(layer_indices, target_h, target_inter, k_ids, save_name,
                         needs_reduction, apply_pca=False):
            save_path = os.path.join(students_dir, save_name)
            os.makedirs(save_path, exist_ok=True)

            student_hf, tok = create_pruned_student(
                t["model_id"], layer_indices,
                layer_accessor=t["layer_accessor"],
                trust_remote_code=t["trust_remote_code"],
            )
            if needs_reduction:
                if apply_pca and corpus_texts:
                    student_hf = reduce_hidden_dim_pca(
                        student_hf, tok, target_h, corpus_texts,
                        new_intermediate_size=target_inter,
                        trust_remote_code=t["trust_remote_code"],
                    )
                else:
                    student_hf = reduce_hidden_dim(student_hf, target_h, target_inter,
                                                   trust_remote_code=t["trust_remote_code"])

            hf_tmp = os.path.join(save_path, "_hf_pruned")
            student_hf = prune_tokenizer_and_embeddings(student_hf, tok, k_ids, hf_tmp)
            tok = AutoTokenizer.from_pretrained(hf_tmp, trust_remote_code=t["trust_remote_code"])
            save_as_sentence_transformer(student_hf, tok, save_path)
            shutil.rmtree(hf_tmp, ignore_errors=True)

            try:
                st = SentenceTransformer(save_path, trust_remote_code=True)
                embeddings = st.encode(["Hello", "안녕"])
                print(f"  Sanity check: {embeddings.shape}")
                del st
            except Exception as e:
                print(f"  Sanity check failed: {e}")

            return save_path

        # Build intermediate if needed
        if self._needs_two_stage:
            mid_target = teacher_est["total_params"] // 5
            mid_mb = mid_target * 4 / (1024 ** 2)
            mid_opt = find_optimal_config(self.teacher_key, mid_target, mid_mb, min_layers,
                                          corpus_vocab_size=actual_vocab)
            print(f"\n  Building intermediate model (~1/5 teacher)...")
            self._intermediate_path = _build_model(
                mid_opt["layer_indices"], mid_opt["hidden_dim"], mid_opt["intermediate_size"],
                keep_ids, f"{self.teacher_key}_intermediate", mid_opt["needs_hidden_reduction"],
            )

        # Build final
        pca_label = " (PCA)" if use_pca and self._needs_hidden_reduction else ""
        print(f"\n  Building final compressed model{pca_label}...")
        self._student_path = _build_model(
            self.layer_indices, self.hidden_dim, self.intermediate_size,
            keep_ids, f"{self.teacher_key}_compressed", self._needs_hidden_reduction,
            apply_pca=use_pca,
        )

        print(f"\nCompression complete: {self._student_path}")
        return self._student_path

    # ── Distillation ─────────────────────────────────────────────

    def distill(self, epochs: int = 10, batch_size: int = 32, lr: float = 2e-5,
                patience: int = 3, device: str | None = None) -> str:
        """Run knowledge distillation on the created student.

        Automatically uses two-stage if needed.
        """
        from smallmodel.distill import distill as _distill, distill_two_stage

        if self._student_path is None:
            raise ValueError("No student model. Call .create() or .compress() first.")

        students_dir = os.path.join(self.output_dir, "students", self.teacher_key)

        if self._needs_two_stage and self._intermediate_path:
            student_name = os.path.basename(self._student_path)
            self._distilled_path = distill_two_stage(
                self.teacher_key, student_name, students_dir,
                epochs=epochs, batch_size=batch_size, lr=lr,
                device=device, patience=patience,
            )
        else:
            self._distilled_path = _distill(
                teacher_name=self.teacher["model_id"],
                student_path=self._student_path,
                epochs=epochs, batch_size=batch_size, lr=lr,
                device=device, trust_remote_code=self.teacher["trust_remote_code"],
                patience=patience,
            )

        return self._distilled_path

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate(self, task_groups: list[str] | None = None,
                 include_teacher: bool = False) -> dict:
        """Run MTEB evaluation on the student (and optionally teacher).

        Requires: pip install smallmodel[eval]
        """
        import mteb
        import gc
        from sentence_transformers import SentenceTransformer
        import torch

        MTEB_TASK_GROUPS = {
            "Classification": [
                "AmazonCounterfactualClassification", "Banking77Classification",
                "ImdbClassification", "MTOPDomainClassification",
                "MassiveIntentClassification", "MassiveScenarioClassification",
                "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
            ],
            "Clustering": [
                "ArXivHierarchicalClusteringP2P", "ArXivHierarchicalClusteringS2S",
                "BiorxivClusteringP2P.v2", "MedrxivClusteringP2P.v2", "MedrxivClusteringS2S.v2",
                "StackExchangeClustering.v2", "StackExchangeClusteringP2P.v2", "TwentyNewsgroupsClustering.v2",
            ],
            "STS": [
                "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS17",
                "STS22.v2", "STSBenchmark",
            ],
        }

        groups = task_groups or list(MTEB_TASK_GROUPS.keys())
        results_dir = os.path.join(self.output_dir, "results", self.teacher_key)
        os.makedirs(results_dir, exist_ok=True)

        model_paths = []
        if include_teacher:
            model_paths.append(("teacher", self.teacher["model_id"]))
        if self._student_path:
            model_paths.append(("student", self._student_path))
        if self._distilled_path:
            model_paths.append(("distilled", self._distilled_path))

        all_results = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for label, path in model_paths:
            print(f"\nEvaluating: {label} ({path})")
            model = SentenceTransformer(path, trust_remote_code=True, device=device)

            for group in groups:
                tasks = MTEB_TASK_GROUPS.get(group, [])
                for task_name in tasks:
                    try:
                        eval_tasks = mteb.get_tasks(tasks=[task_name])
                        if eval_tasks:
                            evaluation = mteb.MTEB(tasks=eval_tasks)
                            save_path = os.path.join(results_dir, label)
                            evaluation.run(model, output_folder=save_path, eval_splits=["test"])
                    except Exception as e:
                        print(f"    [FAIL] {task_name}: {e}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_results

    # ── Web UI ───────────────────────────────────────────────────

    def serve(self, host: str = "127.0.0.1", port: int = 7860):
        """Launch interactive web UI for model editing.

        Opens a browser where you can:
        - Visualize teacher model layers
        - Select/deselect layers interactively
        - Adjust hidden dim and vocab settings
        - See real-time size estimation
        - Create the student model with one click
        """
        from smallmodel.web.app import create_app

        app = create_app(self)
        print(f"\nSmallModel Web UI: http://{host}:{port}")
        print("Press Ctrl+C to stop.\n")
        app.run(host=host, port=port, debug=False)

    # ── Properties ───────────────────────────────────────────────

    @property
    def student_path(self) -> str | None:
        return self._student_path

    @property
    def distilled_path(self) -> str | None:
        return self._distilled_path

    def __repr__(self) -> str:
        t = self.teacher
        layers = f"{len(self.layer_indices)}L" if self.layer_indices else "auto"
        return (f"SmallModel(teacher='{self.teacher_key}', "
                f"layers={layers}, hidden={self.hidden_dim})")
