"""Flask web app for interactive model editing."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import TYPE_CHECKING

from flask import Flask, render_template, request, jsonify

if TYPE_CHECKING:
    from smallmodel.core import SmallModel

# Distillation dataset definitions
DISTILL_DATASETS = {
    "amazon_counterfactual": {
        "hf_id": "mteb/amazon_counterfactual",
        "label": "Amazon Counterfactual",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train", "test"],
    },
    "banking77": {
        "hf_id": "mteb/banking77",
        "label": "Banking77",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train", "test"],
    },
    "imdb": {
        "hf_id": "mteb/imdb",
        "label": "IMDB",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train", "test"],
    },
    "mtop_domain": {
        "hf_id": "mteb/mtop_domain",
        "label": "MTOP Domain",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train", "test"],
    },
    "massive_intent": {
        "hf_id": "mteb/amazon_massive_intent",
        "label": "MASSIVE Intent (multilingual)",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train"],
        "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"],
    },
    "massive_scenario": {
        "hf_id": "mteb/amazon_massive_scenario",
        "label": "MASSIVE Scenario (multilingual)",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train"],
        "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"],
    },
    "toxic_conversations": {
        "hf_id": "mteb/toxic_conversations_50k",
        "label": "Toxic Conversations",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["test"],
    },
    "tweet_sentiment": {
        "hf_id": "mteb/tweet_sentiment_extraction",
        "label": "Tweet Sentiment",
        "group": "Classification",
        "text_fields": ["text"],
        "splits": ["train", "test"],
    },
    # ── Clustering datasets ──
    "arxiv_clustering_p2p": {
        "hf_id": "mteb/arxiv-clustering-p2p",
        "label": "ArXiv Clustering P2P",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "arxiv_clustering_s2s": {
        "hf_id": "mteb/arxiv-clustering-s2s",
        "label": "ArXiv Clustering S2S",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "biorxiv_clustering_p2p": {
        "hf_id": "mteb/biorxiv-clustering-p2p",
        "label": "BioRxiv Clustering P2P",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "medrxiv_clustering_p2p": {
        "hf_id": "mteb/medrxiv-clustering-p2p",
        "label": "MedRxiv Clustering P2P",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "medrxiv_clustering_s2s": {
        "hf_id": "mteb/medrxiv-clustering-s2s",
        "label": "MedRxiv Clustering S2S",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "stackexchange_clustering": {
        "hf_id": "mteb/stackexchange-clustering",
        "label": "StackExchange Clustering",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "stackexchange_clustering_p2p": {
        "hf_id": "mteb/stackexchange-clustering-p2p",
        "label": "StackExchange Clustering P2P",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    "twentynewsgroups_clustering": {
        "hf_id": "mteb/twentynewsgroups-clustering",
        "label": "TwentyNewsgroups Clustering",
        "group": "Clustering",
        "text_fields": ["sentences"],
        "splits": ["test"],
        "is_cluster": True,
    },
    # ── STS datasets ──
    "biosses": {
        "hf_id": "mteb/biosses-sts",
        "label": "BIOSSES",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sickr": {
        "hf_id": "mteb/SICK-R",
        "label": "SICK-R",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts12": {
        "hf_id": "mteb/sts12-sts",
        "label": "STS12",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts13": {
        "hf_id": "mteb/sts13-sts",
        "label": "STS13",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts14": {
        "hf_id": "mteb/sts14-sts",
        "label": "STS14",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts15": {
        "hf_id": "mteb/sts15-sts",
        "label": "STS15",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts17": {
        "hf_id": "mteb/sts17-crosslingual-sts",
        "label": "STS17 (crosslingual)",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "sts22": {
        "hf_id": "mteb/sts22-crosslingual-sts",
        "label": "STS22 (crosslingual)",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["test"],
    },
    "stsb": {
        "hf_id": "mteb/stsbenchmark-sts",
        "label": "STS Benchmark",
        "group": "STS",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["train", "test"],
    },
}

# MTEB evaluation tasks
MTEB_EVAL_TASKS = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "Banking77Classification",
        "ImdbClassification",
        "MTOPDomainClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
        "TwentyNewsgroupsClustering.v2",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12", "STS13", "STS14", "STS15", "STS17",
        "STS22.v2",
        "STSBenchmark",
    ],
}

# In-memory cache for loaded corpus token analysis
_corpus_cache: dict = {}


def create_app(sm: SmallModel) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/teacher")
    def get_teacher():
        info = sm.get_teacher_info()
        return jsonify(info)

    @app.route("/api/teachers")
    def list_teachers():
        from smallmodel.teachers import TEACHERS
        from smallmodel.sizing import estimate_for_teacher
        result = {}
        for key, t in TEACHERS.items():
            layers = list(range(t["num_layers"]))
            est = estimate_for_teacher(key, layers)
            result[key] = {
                "model_id": t["model_id"],
                "short_name": t["short_name"],
                "num_layers": t["num_layers"],
                "hidden_dim": t["hidden_dim"],
                "intermediate_size": t["intermediate_size"],
                "vocab_size": t["vocab_size"],
                "is_decoder": t.get("is_decoder", False),
                "has_glu": t.get("has_glu", False),
                "total_params": est["total_params"],
                "fp32_mb": est["fp32_mb"],
            }
        return jsonify(result)

    @app.route("/api/estimate", methods=["POST"])
    def estimate():
        data = request.json
        teacher_key = data.get("teacher_key", sm.teacher_key)
        layer_indices = data.get("layer_indices", [])
        hidden_dim = data.get("hidden_dim")
        intermediate_size = data.get("intermediate_size")
        vocab_size = data.get("vocab_size")

        from smallmodel.sizing import estimate_for_teacher
        est = estimate_for_teacher(
            teacher_key, layer_indices,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
        )

        from smallmodel.teachers import TEACHERS
        t = TEACHERS[teacher_key]
        teacher_layers = list(range(t["num_layers"]))
        teacher_est = estimate_for_teacher(teacher_key, teacher_layers)
        compression = teacher_est["total_params"] / max(est["total_params"], 1)

        return jsonify({
            **est,
            "compression_ratio": round(compression, 1),
            "needs_two_stage": compression > 10,
        })

    @app.route("/api/presets")
    def presets():
        teacher_key = request.args.get("teacher_key", sm.teacher_key)
        from smallmodel.teachers import TEACHERS
        from smallmodel.sizing import make_uniform_indices

        t = TEACHERS[teacher_key]
        n = t["num_layers"]
        result = []
        for count in [2, 3, 4, 6, 8]:
            if count < n:
                indices = make_uniform_indices(n, count)
                result.append({
                    "name": f"L{count} Uniform",
                    "layer_indices": indices,
                    "description": f"{count} layers, evenly spaced",
                })
        half = n // 2
        result.append({"name": f"L{half} Top", "layer_indices": list(range(n - half, n)),
                        "description": f"{half} layers, top half (semantic-focused)"})
        result.append({"name": f"L{half} Bottom", "layer_indices": list(range(half)),
                        "description": f"{half} layers, bottom half (syntactic-focused)"})
        result.append({"name": "L2 Ends", "layer_indices": [0, n - 1],
                        "description": "2 layers, first + last (minimal)"})
        return jsonify(result)

    # ── Dataset / Task APIs ──────────────────────────────────────

    @app.route("/api/datasets")
    def list_datasets():
        """Return available distillation datasets grouped by category."""
        result = {}
        for key, ds in DISTILL_DATASETS.items():
            result[key] = {
                "label": ds["label"],
                "group": ds["group"],
                "hf_id": ds["hf_id"],
            }
        return jsonify(result)

    @app.route("/api/eval-tasks")
    def list_eval_tasks():
        """Return MTEB evaluation tasks grouped by category."""
        return jsonify(MTEB_EVAL_TASKS)

    @app.route("/api/vocab-analysis", methods=["POST"])
    def vocab_analysis():
        """Analyze vocab coverage for selected datasets and vocab size.

        Downloads tokenizer + datasets from HuggingFace Hub on first use.
        Results are cached in memory for subsequent calls.
        """
        data = request.json
        teacher_key = data.get("teacher_key", sm.teacher_key)
        dataset_keys = data.get("datasets", list(DISTILL_DATASETS.keys()))
        target_vocab = data.get("target_vocab")

        cache_key = f"{teacher_key}:{'|'.join(sorted(dataset_keys))}"

        if cache_key not in _corpus_cache:
            from smallmodel.teachers import TEACHERS
            from transformers import AutoTokenizer

            t = TEACHERS[teacher_key]

            # Step 1: Download tokenizer from HuggingFace
            print(f"[vocab-analysis] Downloading tokenizer: {t['model_id']}...")
            tokenizer = AutoTokenizer.from_pretrained(
                t["model_id"], trust_remote_code=t["trust_remote_code"]
            )
            print(f"[vocab-analysis] Tokenizer ready (vocab: {t['vocab_size']:,})")

            # Step 2: Download & load dataset texts from HuggingFace
            print(f"[vocab-analysis] Loading {len(dataset_keys)} datasets from HuggingFace...")
            texts = _load_dataset_texts(dataset_keys, max_per_dataset=5000)
            print(f"[vocab-analysis] Loaded {len(texts):,} texts")

            if not texts:
                return jsonify({
                    "error": "No texts loaded. Check your internet connection or dataset selection.",
                    "total_tokens": 0,
                    "unique_tokens": 0,
                    "original_vocab": t["vocab_size"],
                    "total_texts": 0,
                    "special_count": 0,
                    "coverage_curve": [],
                    "coverage_at_target": None,
                    "target_vocab": target_vocab,
                }), 200

            # Step 3: Tokenize and count frequencies
            print(f"[vocab-analysis] Tokenizing {len(texts):,} texts...")
            freq = Counter()
            batch_size = 500
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, add_special_tokens=True,
                                    truncation=True, max_length=128)
                for ids in encoded["input_ids"]:
                    freq.update(ids)

            total_freq = sum(freq.values())
            special_ids = set(tokenizer.all_special_ids)
            corpus_unique = len(freq)
            original_vocab = t["vocab_size"]

            # Build cumulative coverage curve
            sorted_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
            cumsum = 0
            coverage_points = []
            for i, tid in enumerate(sorted_tokens):
                cumsum += freq[tid]
                count = i + 1
                pct = cumsum / total_freq * 100
                if count <= 100 or count % 500 == 0 or pct >= 99.99:
                    coverage_points.append({"vocab": count, "coverage": round(pct, 2)})
                if pct >= 99.999:
                    break

            _corpus_cache[cache_key] = {
                "total_tokens": total_freq,
                "unique_tokens": corpus_unique,
                "original_vocab": original_vocab,
                "total_texts": len(texts),
                "coverage_curve": coverage_points,
                "freq": freq,
                "special_count": len(special_ids),
            }
            print(f"[vocab-analysis] Done: {corpus_unique:,} unique tokens / {total_freq:,} total")

        cached = _corpus_cache[cache_key]

        # Compute coverage for specific target vocab
        coverage_at_target = None
        if target_vocab is not None:
            freq = cached["freq"]
            total_freq = cached["total_tokens"]
            sorted_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
            cumsum = 0
            for i, tid in enumerate(sorted_tokens):
                cumsum += freq[tid]
                if i + 1 >= target_vocab:
                    break
            coverage_at_target = round(cumsum / total_freq * 100, 2)

        return jsonify({
            "total_tokens": cached["total_tokens"],
            "unique_tokens": cached["unique_tokens"],
            "original_vocab": cached["original_vocab"],
            "total_texts": cached["total_texts"],
            "special_count": cached["special_count"],
            "coverage_curve": cached["coverage_curve"],
            "coverage_at_target": coverage_at_target,
            "target_vocab": target_vocab,
        })

    # ── Model creation / compression ─────────────────────────────

    @app.route("/api/create", methods=["POST"])
    def create_model():
        data = request.json
        teacher_key = data.get("teacher_key", sm.teacher_key)
        layer_indices = data.get("layer_indices", [])
        hidden_dim = data.get("hidden_dim")
        vocab_size = data.get("vocab_size")
        name = data.get("name", f"{teacher_key}_custom")

        from smallmodel.teachers import TEACHERS
        sm.teacher_key = teacher_key
        sm.teacher = TEACHERS[teacher_key].copy()
        sm.layer_indices = layer_indices
        if hidden_dim:
            sm.hidden_dim = hidden_dim
        if vocab_size:
            sm.max_vocab = vocab_size

        try:
            path = sm.create(name=name)
            return jsonify({"status": "ok", "path": path})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/compress", methods=["POST"])
    def compress_model():
        data = request.json
        teacher_key = data.get("teacher_key", sm.teacher_key)
        max_params = data.get("max_params", 20_000_000)
        max_mb = data.get("max_fp32_mb", 50.0)
        min_layers = data.get("min_layers", 4)

        from smallmodel.teachers import TEACHERS
        sm.teacher_key = teacher_key
        sm.teacher = TEACHERS[teacher_key].copy()

        try:
            path = sm.compress(
                max_params=max_params, max_fp32_mb=max_mb, min_layers=min_layers,
            )
            return jsonify({
                "status": "ok",
                "path": path,
                "layer_indices": sm.layer_indices,
                "hidden_dim": sm.hidden_dim,
                "intermediate_size": sm.intermediate_size,
                "needs_two_stage": sm._needs_two_stage,
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # ── Distillation (SSE streaming) ────────────────────────────

    _distill_progress: dict = {}  # session_id -> DistillProgress

    @app.route("/api/distill/start", methods=["POST"])
    def distill_start():
        """Start distillation in a background thread."""
        import threading
        from smallmodel.web.distill_runner import DistillProgress, run_distillation

        data = request.json
        teacher_path = data.get("teacher_path", "")
        student_path = data.get("student_path", "")
        output_path = data.get("output_path", "")
        dataset_keys = data.get("datasets", list(DISTILL_DATASETS.keys()))
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 32)
        lr = data.get("lr", 2e-5)
        patience = data.get("patience", 3)
        device = data.get("device", "cpu")
        cos_weight = data.get("cos_weight", 0.5)
        mse_weight = data.get("mse_weight", 1.0)
        max_length = data.get("max_length", 64)
        save_every_epoch = data.get("save_every_epoch", True)

        # Resolve teacher path from key
        if teacher_path in TEACHERS_MAP:
            from smallmodel.teachers import TEACHERS
            t = TEACHERS[teacher_path]
            trust_teacher = t.get("trust_remote_code", False)
            teacher_path = t["model_id"]
        else:
            trust_teacher = False

        if not teacher_path or not student_path:
            return jsonify({"status": "error", "message": "teacher_path and student_path required"}), 400

        if not output_path:
            output_path = student_path + "_distilled"

        # Load dataset texts
        print(f"[distill] Loading {len(dataset_keys)} datasets...")
        texts = _load_dataset_texts(dataset_keys, max_per_dataset=5000)
        if not texts:
            return jsonify({"status": "error", "message": "No texts loaded from datasets"}), 400
        print(f"[distill] Loaded {len(texts):,} texts")

        # Create progress tracker
        session_id = f"distill_{int(time.time())}"
        progress = DistillProgress()
        _distill_progress[session_id] = progress

        # Start background thread
        thread = threading.Thread(
            target=run_distillation,
            kwargs={
                "teacher_path": teacher_path,
                "student_path": student_path,
                "output_path": output_path,
                "texts": texts,
                "progress": progress,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "cos_weight": cos_weight,
                "mse_weight": mse_weight,
                "patience": patience,
                "device": device,
                "trust_remote_code_teacher": trust_teacher,
                "save_every_epoch": save_every_epoch,
            },
            daemon=True,
        )
        thread.start()

        return jsonify({"status": "ok", "session_id": session_id})

    @app.route("/api/distill/stream/<session_id>")
    def distill_stream(session_id):
        """SSE endpoint for real-time distillation progress."""
        from flask import Response

        progress = _distill_progress.get(session_id)
        if not progress:
            return jsonify({"error": "session not found"}), 404

        return Response(
            progress.stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/distill/stop/<session_id>", methods=["POST"])
    def distill_stop(session_id):
        """Request early stop for a running distillation."""
        progress = _distill_progress.get(session_id)
        if progress and progress.running:
            progress.finished = True
            return jsonify({"status": "ok"})
        return jsonify({"status": "not_running"})

    @app.route("/api/models")
    def list_models():
        """List locally available student models for distillation source/target."""
        import glob as _glob

        models = []
        students_base = os.path.join(sm.output_dir, "students")
        if os.path.isdir(students_base):
            for teacher_dir in os.listdir(students_base):
                teacher_path = os.path.join(students_base, teacher_dir)
                if not os.path.isdir(teacher_path):
                    continue
                for model_name in os.listdir(teacher_path):
                    model_path = os.path.join(teacher_path, model_name)
                    if not os.path.isdir(model_path):
                        continue
                    # Check if it has model files
                    has_model = any(
                        os.path.exists(os.path.join(model_path, f))
                        for f in ["config.json", "modules.json"]
                    )
                    if has_model:
                        size_mb = 0
                        for fname in ["model.safetensors", "pytorch_model.bin"]:
                            for p in [model_path, os.path.join(model_path, "0_Transformer")]:
                                fp = os.path.join(p, fname)
                                if os.path.exists(fp):
                                    size_mb = os.path.getsize(fp) / (1024**2)
                                    break
                        models.append({
                            "name": f"{teacher_dir}/{model_name}",
                            "path": model_path,
                            "size_mb": round(size_mb, 1),
                        })

        # Also list registered teachers as sources
        from smallmodel.teachers import TEACHERS
        teacher_list = []
        for key, t in TEACHERS.items():
            teacher_list.append({
                "key": key,
                "model_id": t["model_id"],
                "short_name": t["short_name"],
            })

        return jsonify({"local_models": models, "teachers": teacher_list})

    @app.route("/api/device-info")
    def device_info():
        """Return available compute devices."""
        import torch
        devices = [{"id": "cpu", "name": "CPU"}]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                devices.append({
                    "id": f"cuda:{i}" if i > 0 else "cuda",
                    "name": f"GPU {i}: {name} ({mem:.1f}GB)",
                })
        return jsonify({"devices": devices})

    # ── HuggingFace Upload ───────────────────────────────────────

    @app.route("/api/upload", methods=["POST"])
    def upload_to_hub():
        """Upload a model to HuggingFace Hub."""
        data = request.json
        model_path = data.get("model_path", "")
        repo_id = data.get("repo_id", "")
        hf_token = data.get("hf_token", "")

        if not model_path or not repo_id or not hf_token:
            return jsonify({"status": "error", "message": "model_path, repo_id, hf_token required"}), 400

        if not os.path.isdir(model_path):
            return jsonify({"status": "error", "message": f"Model not found: {model_path}"}), 400

        try:
            from huggingface_hub import HfApi, create_repo, upload_folder

            api = HfApi(token=hf_token)
            create_repo(repo_id, token=hf_token, exist_ok=True)
            upload_folder(
                repo_id=repo_id,
                folder_path=model_path,
                token=hf_token,
                commit_message=f"Upload model from SmallModel web UI",
            )
            return jsonify({
                "status": "ok",
                "url": f"https://huggingface.co/{repo_id}",
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # Helper for teacher lookup
    TEACHERS_MAP = set()
    try:
        from smallmodel.teachers import TEACHERS as _T
        TEACHERS_MAP = set(_T.keys())
    except Exception:
        pass

    return app


def _load_dataset_texts(dataset_keys: list[str], max_per_dataset: int = 5000) -> list[str]:
    """Load texts from selected datasets (downloads from HuggingFace Hub)."""
    from datasets import load_dataset

    texts = []
    for idx, key in enumerate(dataset_keys, 1):
        if key not in DISTILL_DATASETS:
            continue
        ds_config = DISTILL_DATASETS[key]
        hf_id = ds_config["hf_id"]
        text_fields = ds_config["text_fields"]
        splits = ds_config["splits"]
        subsets = ds_config.get("subsets", [None])
        is_cluster = ds_config.get("is_cluster", False)

        print(f"  [{idx}/{len(dataset_keys)}] Loading {ds_config['label']} ({hf_id})...")
        for subset in subsets:
            for split in splits:
                try:
                    ds = load_dataset(hf_id, subset, split=split) if subset else load_dataset(hf_id, split=split)
                    count = 0

                    if is_cluster:
                        # Clustering datasets: "sentences" field is a list of strings per row
                        for row in ds:
                            for field in text_fields:
                                val = row.get(field, [])
                                if isinstance(val, list):
                                    for text in val:
                                        if text and len(str(text)) > 5:
                                            texts.append(str(text))
                                            count += 1
                                            if count >= max_per_dataset:
                                                break
                                elif val and len(str(val)) > 5:
                                    texts.append(str(val))
                                    count += 1
                            if count >= max_per_dataset:
                                break
                    else:
                        for row in ds:
                            for field in text_fields:
                                text = row.get(field, "")
                                if text and len(str(text)) > 5:
                                    texts.append(str(text))
                                    count += 1
                                    if count >= max_per_dataset:
                                        break
                            if count >= max_per_dataset:
                                break
                except Exception:
                    pass

    return texts
