# vindex-infer — Command reference

Complete guide to every CLI entry point, flag, and Python helper in this repository. All paths are examples; adjust for your machine.

**Binary location after build**

| Platform | Path |
|----------|------|
| Linux / macOS | `./target/release/vindex-infer` |
| Windows | `.\target\release\vindex-infer.exe` |

---

## Build

```bash
# CPU only (no Vulkan SDK required at runtime)
cargo build --release --no-default-features

# CPU + Vulkan GPU backend
cargo build --release
```

Requires **Rust 1.75+**. For GPU builds you need a Vulkan driver on the machine (no CUDA).

---

## Rust binary: subcommands overview

One executable exposes several **subcommands** (first argument after the binary name):

| Subcommand | Purpose |
|------------|---------|
| *(default — no subcommand)* | Run inference, conversation, grounded-causal audit, or A-Vindex probes |
| `check-hf-model` | Verify a Hugging Face folder has `config.json` + `*.safetensors` |
| `download-hf-model` | Download HF weights via `hf download` (needs `hf` on PATH) |
| `extract-attn-avindex` | Build A-Vindex sidecar files next to a vindex from HF weights |

```bash
# Help (if your build supports it)
./target/release/vindex-infer --help
```

---

## 1. Main inference (default)

**What it does:** Loads a `.vindex` directory (mmap’d weights), runs a forward pass, prints **top-K next-token** predictions (or greedy multi-token output with `--conversation`). Supports full attention+FFN, FFN-only ablation, and grounded causal head pruning.

### Required

| Flag | Description |
|------|-------------|
| `--vindex PATH` / `-v` | Directory containing `index.json`, FFN blobs, embeddings, etc. |

### Input (pick one)

| Flag | Description |
|------|-------------|
| `--prompt TEXT` / `-p` | Text prompt; needs `tokenizer.json` under vindex or `--tokenizer` |
| `--token-ids "a,b,c"` | Comma-separated token IDs; **overrides** `--prompt` |
| *(none)* | Default test prompt: *The capital of France is* (Gemma 3 IDs) |

BOS token **2** is prepended automatically if missing.

### Forward modes (`--forward`)

| Value | Needs `attn_weights.bin` | Behavior |
|-------|--------------------------|----------|
| `full` *(default)* | **Yes** (in vindex or `--attn-weights`) | Standard Gemma path: GQA attention + SwiGLU FFN |
| `ffn-only` | No | Skips attention sublayer; **not** HuggingFace-equivalent (diagnostic) |
| `grounded-causal` | **Yes** | **Dual pass:** full reference vs pruned-head causal routing + alignment audit + top-K comparison |

### Grounded causal tuning (only with `--forward grounded-causal`)

| Flag | Default | Description |
|------|---------|-------------|
| `--grounded-energy-threshold` | `0.88` | Cumulative head-probability mass to keep (dynamic K) |
| `--grounded-softmax-temp` | `4.5` | Temperature over per-head energies (avoids winner-take-all collapse) |
| `--grounded-min-k` | `2` | Minimum active heads per layer after layer 0 |

**Incompatible with:** `--conversation` (use conversation + `full` for compare-style audit, or conversation + `grounded-causal` for generation only)

### Conversation (greedy generation)

| Flag | Default | Description |
|------|---------|-------------|
| `--conversation` | off | Repeatedly append **top-1** token until EOS or cap |
| `--max-new-tokens` | `256` | Safety cap on new tokens |

**Requires:** `--prompt` and a tokenizer. **Cannot** use with `--token-ids`.

Works with **`--forward full`**, **`ffn-only`**, or **`grounded-causal`** (grounded uses pruned heads each step; slower than `full` per token).

### Output and performance

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | `10` | Number of next-token candidates to print (single-shot mode) |
| `--gpu N` | CPU | Vulkan device index (only if built **with** default GPU features) |
| `--tokenizer PATH` | `<vindex>/tokenizer.json` | Override tokenizer path |
| `--attn-weights PATH` | inside vindex | Path to `attn_weights.bin` when split from FFN tree |

### A-Vindex diagnostics (optional sidecar)

Requires **`attn_index.json`**, **`attn_centroids.bin`**, **`attn_k_proj_weights.bin`** next to the vindex (build with `extract-attn-avindex`). These files are **not** used for the transformer forward—probe/scan only.

| Flag | Description |
|------|-------------|
| `--attn-vindex-probe` | Cosine match: last-token embedding × stored k_proj → centroid |
| `--attn-probe-layer` | Layer index (`-1` = middle of `layer_bands.knowledge` in `index.json`) |
| `--attn-probe-kv-head` | KV head index (default `0`) |
| `--attn-vindex-scan` | Scan all (layer, kv_head) centroid hits (verbose) |
| `--attn-scan-layers` | Comma-separated layer ids for scan (empty = all) |

### Examples — single forward

```bash
# Top-10 next tokens (token IDs, Paris prompt)
./target/release/vindex-infer --vindex ./gemma3-4b.vindex --token-ids "818,5279,529,7001,563"

# Same via text prompt
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is"

# FFN-only ablation
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "Hello" --forward ffn-only

# Attention weights outside the vindex folder
./target/release/vindex-infer --vindex ./my-vindex --attn-weights /data/attn_weights.bin -p "Hello"

# Grounded causal: reference vs pruned + head alignment report
./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --forward grounded-causal \
  --token-ids "818,5279,529,7001,563" \
  --grounded-energy-threshold 0.88 \
  --grounded-softmax-temp 4.5 \
  --grounded-min-k 2

# Full forward + A-Vindex probe
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is" --attn-vindex-probe
```

### Examples — conversation

```bash
./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --conversation -p "The capital of France is"

./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --conversation -p "Once upon a time" --max-new-tokens 128

# Grounded causal greedy generation
./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --conversation -p "The capital of France is" \
  --forward grounded-causal --max-new-tokens 32

# FFN-only greedy (fast experiment; quality not representative)
./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --conversation -p "Hello" --forward ffn-only
```

### PowerShell (Windows)

```powershell
.\target\release\vindex-infer.exe --vindex .\gemma3-4b.vindex --conversation -p "The sky is blue because"

.\target\release\vindex-infer.exe --vindex .\gemma3-4b.vindex `
  --conversation -p "The capital of France is" `
  --forward grounded-causal --max-new-tokens 32
```

---

## 2. `check-hf-model`

**What it does:** Checks that `--model-dir` contains Hugging Face `config.json` and at least one `*.safetensors`. Exit **0** if OK, **1** if incomplete (prints diagnosis).

```bash
./target/release/vindex-infer check-hf-model --model-dir /path/to/gemma-3-4b-it
```

| Flag | Required | Description |
|------|----------|-------------|
| `--model-dir` | yes | HF model directory |

---

## 3. `download-hf-model`

**What it does:** Downloads a model snapshot with the Hugging Face CLI (`hf download`). Needs **`hf` on PATH** (`pip install huggingface_hub`).

```bash
# Default target: ./hf_checkout/<repo>/ under current directory
./target/release/vindex-infer download-hf-model --hf-repo google/gemma-3-4b-it

# Explicit destination
./target/release/vindex-infer download-hf-model \
  --hf-repo google/gemma-3-4b-it \
  --model-dir ./gemma3-4b-hf
```

| Flag | Required | Description |
|------|----------|-------------|
| `--hf-repo` | yes | Hugging Face model id |
| `--model-dir` | no | Local directory for `hf download --local-dir` |
| `--hf-token` | no | Token for gated models (or env `HF_TOKEN`) |

**Gated models (e.g. Gemma):** Accept the license on the model page, then `hf login` or set `HF_TOKEN`. Full Gemma 3 4B needs **~9–10+ GB** free disk.

---

## 4. `extract-attn-avindex`

**What it does:** Builds the **A-Vindex attention sidecar** from original HF weights (Lloyd k-means on projected vocab rows per KV head). Writes next to `--vindex`:

- `attn_index.json`
- `attn_centroids.bin`
- `attn_k_proj_weights.bin`

These files enable `--attn-vindex-probe` / `--attn-vindex-scan`; they **do not** replace `attn_weights.bin` for inference.

```bash
# From an existing HF folder
./target/release/vindex-infer extract-attn-avindex \
  --model-dir /path/to/gemma-3-4b-it \
  --vindex ./gemma3-4b.vindex

# Download + extract in one step
./target/release/vindex-infer extract-attn-avindex \
  --hf-repo google/gemma-3-4b-it \
  --vindex ./gemma3-4b.vindex

# f32 sidecar (Python-compatible layout), overwrite existing
./target/release/vindex-infer extract-attn-avindex \
  --model-dir ./gemma3-4b-hf \
  --vindex ./gemma3-4b.vindex \
  --output-dtype f32 \
  --force
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vindex` / `-v` | — | Vindex directory (output sidecar location) |
| `--model-dir` | — | HF folder with `config.json` + safetensors |
| `--hf-repo` | — | If set and model dir missing/incomplete, download first |
| `--hf-token` | env `HF_TOKEN` | Token for gated repos |
| `--centroids` | `256` | Centroids per KV head |
| `--vocab-sample` | `8000` | Vocab rows sampled for k-means |
| `--seed` | `42` | RNG seed |
| `--kmeans-iterations` | `25` | Lloyd iterations |
| `--output-dtype` | `f16` | `f16` or `f32` on-disk for centroids + k_proj |
| `--force` | false | Overwrite existing sidecar files |

---

## 5. Python scripts (`scripts/`)

Install dependencies from the repo:

```bash
cd scripts
pip install -r requirements.txt
```

| Script | Purpose |
|--------|---------|
| `download_vindex_from_hf.py` | Download a pre-extracted vindex from Hugging Face Hub |
| `vindex_infer_python.py` | Full forward in NumPy (port of Rust inference) |
| `vindex_infer_ffn_att.py` | Same + optional **`attn_meta`** summary after forward |
| `build_attn_semantic_meta.py` | Build **Option C** `attn_meta.bin` from `attn_weights.bin` |
| `vindex_causal_grounded.py` | Grounded causal pruning experiment (edit constants in file) |

### `download_vindex_from_hf.py`

**What it does:** `snapshot_download` of a Hub repo into a local folder (~7.3 GB for Gemma 3 4B vindex).

```bash
cd scripts
python download_vindex_from_hf.py
python download_vindex_from_hf.py --repo-id cronos3k/gemma-3-4b-it-vindex --local-dir ../gemma3-4b.vindex
python download_vindex_from_hf.py --token YOUR_HF_TOKEN
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | `cronos3k/gemma-3-4b-it-vindex` | Hub repo |
| `--local-dir` | `gemma3-4b.vindex` | Destination folder |
| `--token` | env / cached login | HF token for gated repos |

### `vindex_infer_python.py`

**What it does:** Faithful **attention + FFN** forward over LarQL mmap binaries; tied LM head on `embeddings.bin`.

```bash
cd scripts
python vindex_infer_python.py --vindex ../gemma3-4b.vindex --token-ids 818,5279,529,7001,563
python vindex_infer_python.py --vindex ../gemma3-4b.vindex -p "The capital of France is"
python vindex_infer_python.py --vindex ../gemma3-4b.vindex -p "Hello" --forward ffn-only
python vindex_infer_python.py --vindex ../my.vindex --attn-weights /data/attn_weights.bin -p "Hi"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vindex` | required | Vindex directory |
| `--attn-weights` | in vindex | External `attn_weights.bin` |
| `-p` / `--prompt` | — | Text input |
| `--token-ids` | — | Comma-separated IDs (overrides prompt) |
| `--tokenizer` | `<vindex>/tokenizer.json` | Tokenizer path |
| `--top-k` | `10` | Next-token candidates |
| `--forward` | `full` | `full` or `ffn-only` |
| `--logits-chunk` | `4096` | Vocab chunk size for logits (memory) |
| `-v` / `--verbose` | off | DEBUG logs |
| `-q` / `--quiet` | off | WARNING only |

### `vindex_infer_ffn_att.py`

**What it does:** Same engine as `vindex_infer_python.py`, plus optional display of **semantic head labels** from `attn_meta.bin` (built separately; **not** used in the forward).

```bash
python vindex_infer_ffn_att.py --vindex ../gemma3-4b.vindex -p "Hello" --attn-meta
python vindex_infer_ffn_att.py --vindex ../gemma3-4b.vindex -p "Hello" \
  --attn-meta --attn-meta-layer 20 --attn-meta-max-heads 16
```

| Flag | Default | Description |
|------|---------|-------------|
| *(same as `vindex_infer_python.py`)* | | |
| `--attn-meta` | off | Print Option C labels after inference |
| `--attn-meta-layer` | `-1` | Layer to show (`-1` = last layer) |
| `--attn-meta-max-heads` | `8` | Max query heads listed |

### `build_attn_semantic_meta.py`

**What it does:** Reads **`attn_weights.bin`** + **`embeddings.bin`**, projects O-slices onto vocab, writes **`attn_meta.bin`** (and optionally scores). Requires **PyTorch**.

```bash
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --top-k 10
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --top-k 20 --cpu
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --no-scores
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vindex` | required | Vindex with `attn_weights.bin` |
| `--top-k` | `10` | Top tokens per (layer, head) |
| `--cpu` | off | Force CPU even if CUDA available |
| `--no-scores` | off | Skip `attn_meta_scores.bin` |

### `vindex_causal_grounded.py`

**What it does:** Python reference for **grounded causal head pruning** (dual reference vs Markov pass, alignment audit). Edit constants at the top of the file (`VINDEX_DIR`, `TEST_TOKENS`, `ENERGY_THRESHOLD`, etc.). Run from `scripts/` so `vindex_infer_python` imports.

```bash
cd scripts
python vindex_causal_grounded.py
```

**Rust equivalent:** `--forward grounded-causal` on the main binary (no separate Python CLI).

---

## 6. LarQL extraction (external)

**What it does:** Creates a vindex from a Hugging Face model (not part of this repo’s binary). Use when you need your own extract or missing `attn_weights.bin`.

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

| Level | Approx. size | Capabilities |
|-------|--------------|--------------|
| `browse` | ~3 GB | Browse / graph queries |
| `inference` | ~6 GB | + inference weights including attention |
| `all` | ~10 GB | + compile-level artifacts |

For **full** `vindex-infer --forward full`, you need **`attn_weights.bin`** (inference or `all` level).

---

## 7. Pre-tokenized test prompts (Gemma 3 4B)

Use with `--token-ids` (BOS `2` added automatically if omitted):

| Prompt | Token IDs |
|--------|-----------|
| The capital of France is → **Paris** | `818,5279,529,7001,563` |
| The largest planet is → **Jupiter** | `818,7488,13401,563` |
| The color of the sky is → **blue** | `818,2258,529,506,7217,563` |
| Einstein was born in → **Ulm** | `147505,691,8132,528` |
| The currency of the UK is → **Pound** | `818,15130,529,506,6322,563` |

---

## 8. Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `HF_TOKEN` | `download-hf-model`, `extract-attn-avindex`, Hub scripts | Hugging Face token for gated models |
| `RUST_LOG` | Rust binary | Log level, e.g. `info`, `vindex_infer=debug` |

**PowerShell:**

```powershell
$env:HF_TOKEN = "hf_..."
$env:RUST_LOG = "vindex_infer=debug"
```

**Unix:**

```bash
export HF_TOKEN=hf_...
export RUST_LOG=vindex_infer=debug
```

---

## 9. Which files matter for what

| File | Used by forward? | Used for |
|------|------------------|----------|
| `gate_vectors.bin`, `up_weights.bin`, `down_weights.bin` | Yes | FFN |
| `attn_weights.bin` | Yes (`full`, `grounded-causal`) | Q/K/V/O attention |
| `norms.bin`, `embeddings.bin`, `index.json` | Yes | Norms, LM head, config |
| `attn_meta.bin` | **No** | Python `--attn-meta` labels only |
| `attn_index.json`, `attn_centroids.bin`, `attn_k_proj_weights.bin` | **No** | `--attn-vindex-probe` / `--attn-vindex-scan` only |

---

## 10. Quick workflows

**A. Download Hub vindex and run inference**

```bash
cd scripts && pip install -r requirements.txt
python download_vindex_from_hf.py --local-dir ../gemma3-4b.vindex
cd .. && cargo build --release --no-default-features
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is"
```

**B. Build semantic attention metadata, then inspect**

```bash
cd scripts
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --top-k 20
python vindex_infer_ffn_att.py --vindex ../gemma3-4b.vindex -p "Hello" --attn-meta --attn-meta-layer 20
```

**C. HF weights → A-Vindex sidecar → probe**

```bash
./target/release/vindex-infer download-hf-model --hf-repo google/gemma-3-4b-it --model-dir ./gemma3-4b-hf
./target/release/vindex-infer extract-attn-avindex --model-dir ./gemma3-4b-hf --vindex ./gemma3-4b.vindex
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is" --attn-vindex-probe
```

**D. Compare full vs grounded causal routing**

```bash
./target/release/vindex-infer --vindex ./gemma3-4b.vindex \
  --forward grounded-causal \
  --token-ids "818,5279,529,7001,563"
```

---

## See also

- [README.md](README.md) — project overview, architecture, verified results  
- [scripts/README.md](scripts/README.md) — Python tools in more detail  
- [gemma3-4b.vindex/README.md](gemma3-4b.vindex/README.md) — Hub snapshot layout (if present locally)
