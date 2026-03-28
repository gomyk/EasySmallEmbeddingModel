# EasySmallEmbeddingModel

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
pip install EasySmallEmbeddingModel[all]
```

Or install specific extras:

```bash
pip install EasySmallEmbeddingModel          # core only
pip install EasySmallEmbeddingModel[web]     # + Flask web UI
pip install EasySmallEmbeddingModel[eval]    # + MTEB evaluation
pip install EasySmallEmbeddingModel[export]  # + ONNX export
pip install EasySmallEmbeddingModel[hub]     # + HuggingFace Hub upload
```

For development:

```bash
git clone https://github.com/gomyk/EasySmallEmbeddingModel.git
cd EasySmallEmbeddingModel
pip install -e ".[all]"
```

---

## Quick Start

### 1. Auto Compress (Easiest)

The simplest way to create a small model. Just specify size constraints and everything is handled automatically.

```python
from smallmodel import SmallModel

sm = SmallModel.from_teacher("gte")
sm.compress(max_fp32_mb=50.0)    # creates optimally compressed student
sm.distill(epochs=10)            # knowledge distillation
```

### 2. Manual Layer Selection

Pick exactly which layers to keep from the teacher model.

```python
from smallmodel import SmallModel

sm = SmallModel.from_teacher("gte", layer_indices=[0, 3, 6, 11])
sm.create()                      # layer pruning + vocab pruning
sm.distill(epochs=10)            # knowledge distillation
```

### 3. Interactive Web UI

```bash
smallmodel serve --teacher gte --port 7860
```

Opens a browser-based editor at `http://127.0.0.1:7860` where you can visually select layers, adjust dimensions, and create models.

---

## Full API Reference

### `SmallModel` Class

The main entry point for all operations.

#### `SmallModel.from_teacher(teacher_key, **kwargs)`

Create a SmallModel instance from a pre-registered teacher.

```python
from smallmodel import SmallModel

# Basic - uses default settings
sm = SmallModel.from_teacher("gte")

# With manual layer selection
sm = SmallModel.from_teacher("gte", layer_indices=[0, 3, 6, 11])

# With custom output directory
sm = SmallModel.from_teacher("gte", output_dir="./my_output")

# With vocab control
sm = SmallModel.from_teacher("gte", vocab_keep_ratio=0.95)   # keep top 95% frequent tokens
sm = SmallModel.from_teacher("gte", max_vocab=50000)          # keep top 50K tokens
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `teacher_key` | `str` | required | Key from teacher registry (e.g. `"gte"`, `"minilm"`) |
| `layer_indices` | `list[int]` | `None` | Which layers to keep. `None` = decided by `.compress()` |
| `hidden_dim` | `int` | teacher's dim | Target hidden dimension. Smaller = smaller model |
| `intermediate_size` | `int` | teacher's size | Target FFN intermediate dimension |
| `vocab_keep_ratio` | `float` | `None` | Keep top N% tokens by cumulative frequency (0.0~1.0) |
| `max_vocab` | `int` | `None` | Hard limit on vocab size |
| `output_dir` | `str` | `"output"` | Where to save student models and results |

---

#### `sm.compress(**kwargs)`

Automatically find the optimal model configuration within size constraints and create the student.

```python
sm = SmallModel.from_teacher("gte")

# Default: 50MB / 20M params
path = sm.compress()

# Custom constraints
path = sm.compress(
    max_fp32_mb=30.0,         # max 30MB (FP32)
    max_params=10_000_000,    # max 10M parameters
    min_layers=4,             # at least 4 layers
    vocab_percentile=0.95,    # keep tokens covering 95% of corpus frequency
    min_vocab=5000,           # but at least 5K vocab
)

print(f"Student saved to: {path}")
print(f"Layers: {sm.layer_indices}")
print(f"Hidden dim: {sm.hidden_dim}")
print(f"Needs 2-stage distillation: {sm._needs_two_stage}")
```

**How it works internally:**
1. Loads a multilingual corpus and analyzes token frequencies
2. Searches for the best (layers, hidden_dim, vocab) combination
3. Priority: preserve hidden_dim > maximize layers > maximize vocab
4. If compression > 10x, also creates an intermediate model for 2-stage distillation
5. Applies layer pruning, hidden dim reduction (if needed), and vocab pruning

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_params` | `int` | `20_000_000` | Maximum total parameters |
| `max_fp32_mb` | `float` | `50.0` | Maximum model size in MB (FP32, 4 bytes/param) |
| `min_layers` | `int` | `4` | Minimum number of transformer layers |
| `vocab_percentile` | `float` | `0.95` | Cumulative frequency threshold for vocab pruning |
| `min_vocab` | `int` | `None` | Minimum vocab size floor |

**Returns:** `str` - Path to the saved student model.

---

#### `sm.create(name=None, no_prune=False)`

Create a student with manually specified layer indices. You must set `layer_indices` first.

```python
sm = SmallModel.from_teacher("minilm", layer_indices=[0, 4, 7, 11])

# With vocab pruning (default)
path = sm.create(name="minilm_L4_custom")

# Without vocab pruning
path = sm.create(name="minilm_L4_full_vocab", no_prune=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `None` | Model directory name. Auto-generated if `None` |
| `no_prune` | `bool` | `False` | Skip vocab pruning (keep all original tokens) |

**Returns:** `str` - Path to the saved student model.

---

#### `sm.distill(**kwargs)`

Run knowledge distillation on the created student. The student model must exist (call `.create()` or `.compress()` first).

```python
sm = SmallModel.from_teacher("gte", layer_indices=[0, 4, 8, 11])
sm.create()

# Basic distillation
path = sm.distill()

# Custom training settings
path = sm.distill(
    epochs=15,          # max epochs (early stopping may stop sooner)
    batch_size=64,      # training batch size
    lr=3e-5,            # learning rate
    patience=5,         # early stopping patience
    device="cuda",      # force GPU ("cpu", "cuda", or None for auto)
)

print(f"Distilled model saved to: {path}")
```

**What it does:**
- Loads MTEB task datasets as training data (downloaded from HuggingFace)
- Teacher encodes each batch -> student tries to reproduce the same embeddings
- Loss = MSE + Cosine Similarity between teacher and student outputs
- If teacher/student dimensions differ, a learnable projection layer is added
- If `.compress()` detected 10x+ compression, automatically uses 2-stage distillation

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `10` | Maximum training epochs |
| `batch_size` | `int` | `32` | Training batch size |
| `lr` | `float` | `2e-5` | Learning rate (AdamW) |
| `patience` | `int` | `3` | Early stopping: stop after N epochs without improvement |
| `device` | `str` | `None` | `None` = auto-detect GPU. `"cuda"` or `"cpu"` to force |

**Returns:** `str` - Path to the distilled model (original path + `"_distilled"` suffix).

---

#### `sm.evaluate(**kwargs)`

Run MTEB benchmark evaluation on the student model.

```python
sm = SmallModel.from_teacher("gte", layer_indices=[0, 4, 8, 11])
sm.create()
sm.distill()

# Evaluate all task groups
sm.evaluate()

# Evaluate specific groups only
sm.evaluate(task_groups=["Classification", "STS"])

# Include teacher baseline for comparison
sm.evaluate(include_teacher=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_groups` | `list[str]` | `None` | `["Classification", "Clustering", "STS"]` if `None` |
| `include_teacher` | `bool` | `False` | Also evaluate the teacher for comparison |

---

#### `sm.estimate(layer_indices=None)`

Estimate model size without actually creating it. Useful for exploring configurations.

```python
sm = SmallModel.from_teacher("gte")

# Estimate with different layer counts
for n_layers in [2, 4, 6, 8]:
    indices = [round(i * 11 / (n_layers - 1)) for i in range(n_layers)]
    est = sm.estimate(layer_indices=indices)
    print(f"  L{n_layers}: {est['total_params']/1e6:.1f}M params, {est['fp32_mb']}MB")
```

**Returns:** `dict` with `total_params` (int) and `fp32_mb` (float).

---

#### `sm.serve(host="127.0.0.1", port=7860)`

Launch the interactive web UI.

```python
sm = SmallModel.from_teacher("gte")
sm.serve(port=8080)
```

---

#### `sm.get_teacher_info()`

Get metadata about the current teacher model.

```python
sm = SmallModel.from_teacher("qwen3")
info = sm.get_teacher_info()
print(info)
# {'key': 'qwen3', 'model_id': 'Qwen/Qwen3-0.6B', 'num_layers': 28,
#  'hidden_dim': 1024, 'vocab_size': 151936, 'total_params': ..., ...}
```

---

#### Properties

```python
sm.student_path      # str | None - path to created student
sm.distilled_path    # str | None - path to distilled model
sm.teacher_key       # str - current teacher key
sm.layer_indices     # list[int] | None - selected layer indices
sm.hidden_dim        # int - target hidden dimension
```

---

### Teacher Registry

#### `register_teacher(key, **kwargs)`

Register a custom teacher model so you can use it with SmallModel.

```python
from smallmodel import register_teacher, SmallModel

# Register a custom model
register_teacher(
    "my-bert",
    model_id="my-org/my-bert-base",     # HuggingFace model ID
    short_name="MyBERT",                 # display name
    hidden_dim=768,                       # hidden dimension
    num_layers=12,                        # number of transformer layers
    intermediate_size=3072,               # FFN intermediate dimension
    vocab_size=30522,                     # vocabulary size
    layer_accessor="encoder.layer",       # how to access layers in the model
    tokenizer_type="wordpiece",           # "unigram", "bpe", or "wordpiece"
    trust_remote_code=False,              # set True for custom model code
)

# Now use it
sm = SmallModel.from_teacher("my-bert")
sm.compress()
```

**Required parameters:**

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | Unique identifier for this teacher |
| `model_id` | `str` | HuggingFace model ID (e.g. `"bert-base-uncased"`) |
| `hidden_dim` | `int` | Hidden dimension of the model |
| `num_layers` | `int` | Number of transformer layers |
| `intermediate_size` | `int` | FFN intermediate dimension |
| `vocab_size` | `int` | Vocabulary size |

**Optional parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_name` | `str` | same as key | Display name |
| `layer_accessor` | `str` | `"encoder.layer"` | Dot-path to access layer list (e.g. `"layers"` for decoder models) |
| `tokenizer_type` | `str` | `"unigram"` | Tokenizer type: `"unigram"`, `"bpe"`, `"wordpiece"` |
| `trust_remote_code` | `bool` | `False` | Allow custom model code from HuggingFace |
| `num_attention_heads` | `int` | `None` | For GQA models |
| `num_kv_heads` | `int` | `None` | For GQA models |
| `head_dim` | `int` | `None` | For models with non-standard head dim (e.g. Qwen3) |
| `has_glu` | `bool` | `False` | Model uses SwiGLU/GLU activation (3 FFN matrices) |
| `is_decoder` | `bool` | `False` | Decoder-only model |

---

### Low-Level API

For advanced users who want fine-grained control over each step.

#### Architecture utilities (`smallmodel.arch`)

```python
from smallmodel.arch import (
    create_pruned_student,
    reduce_hidden_dim,
    collect_corpus_tokens,
    prune_tokenizer_and_embeddings,
    save_as_sentence_transformer,
)

# Step 1: Layer pruning
student_model, tokenizer = create_pruned_student(
    "alibaba-NLP/gte-multilingual-base",
    layer_indices=[0, 3, 6, 11],
    layer_accessor="encoder.layer",
    trust_remote_code=True,
)

# Step 2: Hidden dim reduction (optional)
student_model = reduce_hidden_dim(
    student_model,
    new_hidden_dim=384,
    new_intermediate_size=1536,
    trust_remote_code=True,
)

# Step 3: Vocab pruning
corpus_texts = ["Hello world", "안녕하세요", "こんにちは", ...]
keep_ids = collect_corpus_tokens(tokenizer, texts=corpus_texts, vocab_keep_ratio=0.95)
student_model = prune_tokenizer_and_embeddings(
    student_model, tokenizer, keep_ids, save_dir="./tmp_pruned"
)

# Step 4: Save as SentenceTransformer
save_as_sentence_transformer(student_model, tokenizer, "./my_student")
```

#### Size estimation (`smallmodel.sizing`)

```python
from smallmodel.sizing import (
    estimate_size,
    estimate_for_teacher,
    find_optimal_config,
    make_uniform_indices,
)

# Generate evenly-spaced layer indices
indices = make_uniform_indices(num_layers=12, target_count=4)
# [0, 4, 7, 11]

# Estimate size from raw parameters
est = estimate_size(
    layer_indices=[0, 4, 7, 11],
    hidden_dim=768,
    vocab_size=50000,
    intermediate_size=3072,
)
print(f"{est['total_params']/1e6:.1f}M params, {est['fp32_mb']}MB")

# Estimate using teacher defaults
est = estimate_for_teacher("gte", [0, 4, 7, 11], vocab_size=50000)

# Find optimal config within constraints
opt = find_optimal_config(
    "gte",
    max_params=20_000_000,
    max_fp32_mb=50.0,
    min_layers=4,
)
print(opt)
# {'layer_indices': [0, 4, 7, 11], 'hidden_dim': 768,
#  'intermediate_size': 3072, 'target_vocab': 12345,
#  'needs_hidden_reduction': False}
```

#### Distillation (`smallmodel.distill`)

```python
from smallmodel.distill import distill, distill_two_stage

# Single-stage distillation
distilled_path = distill(
    teacher_name="alibaba-NLP/gte-multilingual-base",
    student_path="./my_student",
    epochs=10,
    batch_size=32,
    lr=2e-5,
    patience=3,
    device="cuda",
    trust_remote_code=True,
)
# Saves to ./my_student_distilled

# Two-stage distillation (for 10x+ compression)
distilled_path = distill_two_stage(
    teacher_key="gte",
    student_name="gte_compressed",
    students_dir="./output/students/gte",
    epochs=10,
)
```

---

## End-to-End Examples

### Example 1: Compress GTE to 50MB and distill

```python
from smallmodel import SmallModel

# Create and compress
sm = SmallModel.from_teacher("gte", output_dir="./gte_project")
sm.compress(max_fp32_mb=50.0, min_layers=4)

# Distill with GPU
sm.distill(epochs=10, batch_size=64, device="cuda")

# Check results
print(f"Student: {sm.student_path}")
print(f"Distilled: {sm.distilled_path}")
```

### Example 2: Create multiple students and compare

```python
from smallmodel import SmallModel

teacher = "minilm"

for n_layers in [3, 4, 6]:
    sm = SmallModel.from_teacher(teacher, output_dir="./compare")

    # Generate uniform layer indices
    from smallmodel.sizing import make_uniform_indices
    sm.layer_indices = make_uniform_indices(12, n_layers)

    # Estimate before creating
    est = sm.estimate()
    print(f"L{n_layers}: {est['total_params']/1e6:.1f}M params, {est['fp32_mb']}MB")

    # Create and distill
    sm.create(name=f"minilm_L{n_layers}")
    sm.distill(epochs=5)
```

### Example 3: Custom teacher + full pipeline

```python
from smallmodel import SmallModel, register_teacher

# Register your own model
register_teacher(
    "my-xlm",
    model_id="xlm-roberta-base",
    short_name="XLM-R-base",
    hidden_dim=768,
    num_layers=12,
    intermediate_size=3072,
    vocab_size=250002,
    layer_accessor="encoder.layer",
    tokenizer_type="unigram",
)

# Compress to 30MB
sm = SmallModel.from_teacher("my-xlm")
sm.compress(max_fp32_mb=30.0, max_params=8_000_000, min_layers=3)

# Distill
sm.distill(epochs=10, patience=3)

# Evaluate
sm.evaluate(task_groups=["Classification", "STS"], include_teacher=True)
```

### Example 4: Low-level control

```python
from smallmodel.arch import (
    create_pruned_student,
    collect_corpus_tokens,
    prune_tokenizer_and_embeddings,
    save_as_sentence_transformer,
)
from smallmodel.distill import distill
from sentence_transformers import SentenceTransformer

# 1. Layer pruning
student, tokenizer = create_pruned_student(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    layer_indices=[0, 6, 11],
)

# 2. Vocab pruning with custom corpus
my_corpus = [
    "예약 좀 해줘",
    "Book a table for two",
    "今日はいい天気ですね",
    # ... your domain texts
]
keep_ids = collect_corpus_tokens(tokenizer, texts=my_corpus)
student = prune_tokenizer_and_embeddings(student, tokenizer, keep_ids, "./tmp")

# 3. Save
save_as_sentence_transformer(student, tokenizer, "./my_tiny_model")

# 4. Distill
distill(
    teacher_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    student_path="./my_tiny_model",
    epochs=10,
    device="cuda",
)

# 5. Use
model = SentenceTransformer("./my_tiny_model_distilled", trust_remote_code=True)
embeddings = model.encode(["Hello!", "안녕하세요!"])
print(embeddings.shape)
```

---

## CLI Reference

```bash
# List all available teacher models
smallmodel list-teachers

# Launch web UI
smallmodel serve --teacher gte --port 7860

# Auto-compress a teacher model
smallmodel compress --teacher gte --max-mb 50 --max-params 20000000 --min-layers 4

# Create student with specific layers
smallmodel create --teacher gte --layers 0,3,6,11 --name my_student

# Create without vocab pruning
smallmodel create --teacher minilm --layers 0,4,7,11 --no-prune

# Run knowledge distillation
smallmodel distill --teacher gte --student ./output/students/gte/gte_compressed --epochs 10 --batch-size 64

# Evaluate on MTEB
smallmodel evaluate --teacher gte --student ./output/students/gte/gte_compressed --include-teacher
```

---

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

```
Teacher Model (e.g. GTE 1058MB)
    │
    ├── 1. Layer Pruning ──────── Keep N layers (uniform spacing recommended)
    │
    ├── 2. Hidden Dim Reduction ─ Shrink dimensions if needed (slicing or PCA)
    │
    ├── 3. Vocab Pruning ──────── Remove tokens not seen in training corpus
    │
    ├── 4. Knowledge Distillation  MSE + Cosine loss on teacher embeddings
    │       │
    │       └── (if >10x compression: Teacher → Intermediate → Student)
    │
    └── 5. Evaluation ─────────── MTEB benchmark (Classification, Clustering, STS)

Student Model (e.g. 50MB)
```

## License

Apache-2.0
