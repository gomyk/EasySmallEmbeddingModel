# SmallModel

Compress large embedding models into small, fast students via layer pruning, vocab pruning, hidden dim reduction, and knowledge distillation.

## Features

- **Layer Pruning** - Select which transformer layers to keep
- **Vocab Pruning** - Remove unused tokens based on corpus frequency
- **Hidden Dim Reduction** - Shrink internal dimensions (slicing or PCA)
- **Knowledge Distillation** - MSE + Cosine loss alignment with teacher
- **Auto Compress** - Find optimal config within size constraints
- **2-Stage Distillation** - Progressive distillation for 10x+ compression
- **Interactive Web UI** - Visual layer editor with real-time size estimation
- **MTEB Evaluation** - Benchmark on Classification, Clustering, STS tasks

## Installation

```bash
pip install smallmodel[all]
```

Or install specific extras:

```bash
pip install smallmodel          # core only
pip install smallmodel[web]     # + Flask web UI
pip install smallmodel[eval]    # + MTEB evaluation
pip install smallmodel[export]  # + ONNX export
pip install smallmodel[hub]     # + HuggingFace Hub upload
```

For development:

```bash
git clone https://github.com/gomyk/smallmodel.git
cd smallmodel
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from smallmodel import SmallModel

# Auto-compress within 50MB
sm = SmallModel.from_teacher("gte")
sm.compress(max_fp32_mb=50.0)
sm.distill(epochs=10)

# Manual layer selection
sm = SmallModel.from_teacher("gte", layer_indices=[0, 3, 6, 11])
sm.create()

# Register custom teacher
from smallmodel import register_teacher
register_teacher(
    "my-bert",
    model_id="my-org/my-bert-base",
    short_name="MyBERT",
    hidden_dim=768, num_layers=12,
    intermediate_size=3072, vocab_size=30522,
)
```

### Web UI

```python
from smallmodel import SmallModel

sm = SmallModel.from_teacher("gte")
sm.serve()  # http://127.0.0.1:7860
```

Or via CLI:

```bash
smallmodel serve --teacher gte --port 7860
```

The web UI lets you:
- Select teacher model from 7+ pre-registered models
- Toggle layers on/off with preset configurations
- Adjust hidden dim, FFN size, and vocab size
- See real-time size estimation and compression ratio
- Select distillation datasets and evaluation tasks
- Analyze vocab coverage at different vocab sizes
- Create compressed models with one click

### CLI

```bash
smallmodel list-teachers
smallmodel compress --teacher gte --max-mb 50
smallmodel create --teacher gte --layers 0,3,6,11
smallmodel distill --teacher gte --student output/students/gte/gte_compressed
smallmodel serve --teacher gte
```

## Pre-registered Teachers

| Key | Model | Layers | Hidden | Vocab | FP32 MB |
|---|---|---|---|---|---|
| minilm | paraphrase-multilingual-MiniLM-L12-v2 | 12 | 384 | 250K | 448 |
| modernbert | ModernBERT-base | 22 | 768 | 50K | 496 |
| gte | gte-multilingual-base | 12 | 768 | 250K | 1058 |
| me5 | multilingual-e5-base | 12 | 768 | 250K | 1058 |
| me5s | multilingual-e5-small | 12 | 384 | 250K | 448 |
| gemma_emb | embeddinggemma-300m | 24 | 768 | 262K | 1155 |
| qwen3 | Qwen3-0.6B | 28 | 1024 | 152K | 2274 |

## How It Works

1. **Layer Pruning** - Copy selected layers from teacher (uniform spacing recommended)
2. **Hidden Dim Reduction** - Shrink dimensions if needed to meet size target
3. **Vocab Pruning** - Remove tokens not seen in training corpus
4. **Knowledge Distillation** - Train student to reproduce teacher's embeddings
5. **Evaluation** - MTEB benchmark (Classification, Clustering, STS)

For compression ratios > 10x, a 2-stage distillation pipeline is used:
Teacher → Intermediate (~1/5 teacher) → Final Student

## License

Apache-2.0
