# vindex-infer

**Vendor-free LLM inference from decomposed weights. CPU or Vulkan GPU. No CUDA required.**

**Paper**: [Framework-Free Transformer Inference from Decomposed Weight Files via Vulkan Compute](https://zenodo.org/records/19609604) (Zenodo)

Run any open-source LLM on any hardware — NVIDIA, AMD, Intel, Apple, or just CPU. No framework dependencies, no vendor lock-in. One Rust binary, flat weight files, exact output.

```bash
# Download pre-extracted model from HuggingFace (7.3 GB)
pip install huggingface-hub
huggingface-cli download cronos3k/gemma-3-4b-it-vindex --local-dir gemma3-4b.vindex

# Build
cargo build --release

# Run inference — CPU (works everywhere, no GPU needed)
./target/release/vindex-infer --vindex gemma3-4b.vindex --token-ids "818,5279,529,7001,563"
#  1. Paris     (+21.24)
#  2. a         (+17.69)
#  3. the       (+17.51)

# FFN-only ablation (not HF-equivalent; diagnostic)
./target/release/vindex-infer --vindex gemma3-4b.vindex -p "The capital of France is" --forward ffn-only

# Build A-Vindex attention sidecar from original HF weights (needs config.json + *.safetensors).
# Sidecar blobs default to f16 on disk; use --output-dtype f32 for full float32 files (Python-compatible).
./target/release/vindex-infer extract-attn-avindex --model-dir /path/to/gemma-3-4b-it --vindex ./gemma3-4b.vindex
./target/release/vindex-infer extract-attn-avindex --model-dir /path/to/gemma-3-4b-it --vindex ./gemma3-4b.vindex --output-dtype f32 --force

# Verify HF weights folder (config.json + at least one *.safetensors); exit 1 if incomplete.
./target/release/vindex-infer check-hf-model --model-dir /path/to/gemma-3-4b-it

# Download HF snapshot into a folder (requires `hf` on PATH: pip install huggingface_hub).
./target/release/vindex-infer download-hf-model --hf-repo google/gemma-3-4b-it
# Optional: --model-dir .\gemma3-4b-hf

# One-shot: download (if needed) then extract. With only --hf-repo, weights go under ./hf_checkout/google__gemma-3-4b-it/ (from cwd).
./target/release/vindex-infer extract-attn-avindex --hf-repo google/gemma-3-4b-it --vindex ./gemma3-4b.vindex

# Or set an explicit folder (recommended on Windows: absolute path or .\relative under your project).
./target/release/vindex-infer extract-attn-avindex --hf-repo google/gemma-3-4b-it --model-dir ./gemma3-4b-hf --vindex ./gemma3-4b.vindex

# Gated models (e.g. Gemma): accept the license on the model’s Hugging Face page first. Then use `hf login`, or set env `HF_TOKEN`, or pass `--hf-token` to `extract-attn-avindex` / `download-hf-model`.

# Disk space: full Gemma 3 4B HF weights need on the order of 9–10+ GB free on the destination drive. A full disk shows as `OSError: [Errno 28] No space left on device` in the `hf` output.

# Logging: default `RUST_LOG=info` shows download + extraction steps. Use `RUST_LOG=vindex_infer=debug` for per–KV-head k-means messages (PowerShell: `$env:RUST_LOG="vindex_infer=debug"`).

# Then full inference + centroid probe
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "Hello" --attn-vindex-probe
```

## Forward modes and conversation (Rust CLI)

**`--forward`** controls each layer’s transformer stack (weights must match what you expect on disk):

| Mode | Flag | `attn_weights.bin` |
|------|------|-------------------|
| **Attention + FFN** (default) | `--forward full` | Required for real attention: file inside **`--vindex`**, or pass **`--attn-weights PATH`** if the blob lives elsewhere next to your FFN/index tree. |
| **FFN only** (ablation) | `--forward ffn-only` | Not used; attention sublayer is skipped (diagnostic only, not HuggingFace-equivalent). |

**`--conversation`** turns on **greedy autoregressive** decoding: each step runs a full forward on the growing sequence and appends the **top-1** token until an **EOS / end-of-turn** token (from the tokenizer) or **`--max-new-tokens`** (default **256**). Requires **`--prompt`** and **`tokenizer.json`** (or **`--tokenizer`**). Cannot be combined with **`--token-ids`**.

Examples (Unix-style paths; on Windows use `.\target\release\vindex-infer.exe` and `.\gemma3-4b.vindex`):

```bash
# One-shot next-token top-k (full stack when attn_weights.bin is present)
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is"

# Same prompt, FFN path only
./target/release/vindex-infer --vindex ./gemma3-4b.vindex -p "The capital of France is" --forward ffn-only

# Full stack but attention weights file is not under the vindex folder
./target/release/vindex-infer --vindex ./my-vindex --attn-weights /data/attn_weights.bin -p "Hello"

# Greedy multi-token generation (full attention + FFN)
./target/release/vindex-infer --vindex ./gemma3-4b.vindex --conversation -p "The capital of France is"
./target/release/vindex-infer --vindex ./gemma3-4b.vindex --conversation -p "Once upon a time" --max-new-tokens 128

# Greedy generation with FFN-only layers (fast experiment; quality is not representative)
./target/release/vindex-infer --vindex ./gemma3-4b.vindex --conversation -p "Hello" --forward ffn-only
```

PowerShell:

```powershell
.\target\release\vindex-infer.exe --vindex .\gemma3-4b.vindex --conversation -p "The sky is blue because"
```

## Pre-extracted model (Hugging Face)

A **ready-made Gemma 3 4B IT** vindex is published on the Hub so you can run **FFN-heavy** workflows and full-stack inference **without** running LarQL locally:

**[cronos3k/gemma-3-4b-it-vindex](https://huggingface.co/cronos3k/gemma-3-4b-it-vindex)** (~7.3 GB, f16, **all** LarQL extraction levels in that snapshot).

### What you get from the Hub

- **FFN path (always the main “pre-extracted” story)** — `gate_vectors.bin`, `up_weights.bin`, `down_weights.bin`, plus `embeddings.bin`, `norms.bin`, `index.json`, tokenizer, etc. That is enough for **FFN-only** diagnostics (`--forward ffn-only`) and for the bulk of the **SwiGLU** stack in a full forward.
- **Full self-attention forward** — the same Hub layout, when downloaded **completely**, includes **`attn_weights.bin`** (packed Q/K/V/O + Q/K norms per layer, f16). Rust and Python inferers use that file for **`--forward full`**. If `attn_weights.bin` is missing (partial copy or browse-level tree), you must add it via **LarQL** (`larql extract-index … --level inference` or `all`) or a full re-download; a Python script cannot invent those tensors.

### Attention *semantic* metadata (Option C) — Python, `build_attn_semantic_meta.py`

The Hub snapshot does **not** ship **`attn_meta.bin`**. For now, that layer is built **locally** by **`scripts/build_attn_semantic_meta.py`**:

1. It **mmap-reads** the same **`attn_weights.bin`** layout LarQL uses (per layer: `W_q | W_k | W_v | W_o | q_norm | k_norm`).
2. For each layer and each **query head**, it derives a direction in hidden space from the **O** projection slice, projects onto the **embedding table** `embeddings.bin` (PyTorch matmul), and records the **top‑*k*** token ids (and optional scores).
3. It writes **`attn_meta.bin`** (uint32), optionally **`attn_meta_scores.bin`**, and merges **`attention_metadata`** into **`index.json`**.

**Requirements:** Python 3.10+, **`pip install -r scripts/requirements.txt`**, and **PyTorch** (GPU optional; use `--cpu` to force CPU).

**Commands** (from repo root; adjust paths):

```bash
cd scripts
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --top-k 10
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --top-k 20 --cpu
python build_attn_semantic_meta.py --vindex ../gemma3-4b.vindex --no-scores   # skip attn_meta_scores.bin
```

After a successful run, inspect labels after inference, e.g.:

```bash
python vindex_infer_ffn_att.py --vindex ../gemma3-4b.vindex -p "Hello" --attn-meta --attn-meta-layer 20
```

More detail on the Python tools lives in **[`scripts/README.md`](scripts/README.md)**.

Or extract your own vindex from any supported model via [LarQL](https://github.com/chrishayuk/larql):

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

## Why This Exists

Every LLM inference solution locks you into a vendor:

| Tool | Requires |
|------|----------|
| PyTorch | Python + CUDA (NVIDIA only) |
| llama.cpp | GGUF format, CUDA for GPU |
| MLX | Apple Silicon only |
| TensorRT | NVIDIA only |
| ONNX Runtime | Microsoft ecosystem |

**vindex-infer requires nothing.** One Rust binary. Flat binary weight files. CPU runs everywhere. GPU runs on Vulkan — which means every discrete GPU, every integrated GPU, every mobile GPU made in the last decade.

## Features

- **Two backends**: CPU (rayon multi-core) and Vulkan GPU (any vendor)
- **Exact output**: Matches HuggingFace Transformers to 5 significant figures
- **Zero framework dependencies**: No Python, no CUDA toolkit, no ML libraries
- **Flat file input**: Vindex format — mmap'd binary files, zero-copy loading
- **Small binary**: ~20 MB compiled, no runtime bloat
- **Forward modes**: **`--forward full`** (attention + FFN when `attn_weights.bin` is available) or **`--forward ffn-only`** (diagnostic)
- **Conversation mode**: **`--conversation`** greedy multi-token generation from **`--prompt`**, optional **`--max-new-tokens`**, with optional **`--attn-weights`** for split layouts
- **Verified**: Paris, Jupiter, blue, Ulm, Pound — factual knowledge retrieval confirmed

## Verified Results

Gemma 3 4B inference (34 layers, 2560 hidden, 262K vocab):

| Prompt | Prediction | Correct? |
|--------|-----------|----------|
| The capital of France is | **Paris** | [YES] |
| The largest planet is | **Jupiter** | [YES] |
| The color of the sky is | **blue** | [YES] |
| Einstein was born in | **Ulm** | [YES] |
| The currency of the UK is | **Pound** (#2) | [YES] |

All predictions match HuggingFace ground truth exactly.

## Performance

| Backend | Time (Gemma 4B) | Hardware |
|---------|----------------|----------|
| CPU (rayon) | ~11s | 64-core AMD EPYC |
| Vulkan GPU | ~4.6s | NVIDIA V100 |

Performance is currently limited by per-operation dispatch overhead (1400+ individual GPU submissions). Batch dispatch optimization is planned.

## Building

```bash
# CPU only (no Vulkan needed)
cargo build --release --no-default-features

# CPU + Vulkan GPU
cargo build --release
```

Requires Rust 1.75+. Vulkan GPU backend requires a Vulkan driver (no SDK needed at runtime).

## Test Prompts (Gemma 3 4B Token IDs)

Since vindex-infer uses token IDs directly (no tokenizer built in yet), here are some pre-tokenized prompts for testing:

```bash
# "The capital of France is" → Paris
--token-ids "818,5279,529,7001,563"

# "The largest planet is" → Jupiter
--token-ids "818,7488,13401,563"

# "The color of the sky is" → blue
--token-ids "818,2258,529,506,7217,563"

# "Einstein was born in" → Ulm
--token-ids "147505,691,8132,528"

# "The currency of the UK is" → Pound (#2)
--token-ids "818,15130,529,506,6322,563"
```

BOS token (2) is prepended automatically.

## How It Works

1. **Vindex loading**: Weight matrices (gate, up, down, attention, embeddings, norms) are memory-mapped from flat binary files. No deserialization — just pointer arithmetic on mmap'd data.

2. **Attention**: Multi-head grouped query attention (GQA) with QK normalization, RoPE positional encoding (HF half-split style), and causal masking.

3. **FFN**: Dense gated projection with GeluTanh activation. Gate and up projections produce activation vectors; down projection maps back to hidden space.

4. **Norms**: RMSNorm with learned gamma weights (offset=1.0 for Gemma 3).

5. **GPU path**: A single Vulkan compute shader (`matvec.comp`) handles all matrix-vector products. Weights are uploaded to VRAM as f16, decoded on-the-fly in the shader. Input/output are f32.

## Model Support

Any model that LarQL can extract to a vindex:

| Family | Models |
|--------|--------|
| Gemma | Gemma 2/3 (2B-27B) |
| Llama | Llama 2/3 (7B-405B) |
| Mistral | Mistral 7B |
| Qwen | Qwen 2/2.5 (0.5B-72B) |
| Mixtral | Mixtral 8x7B, 8x22B |
| DeepSeek | DeepSeek V2/V3 |
| GPT-2 | GPT-2 (117M-1.5B) |

Extract via [LarQL](https://github.com/chrishayuk/larql):
```bash
larql extract-index <model> -o model.vindex --level all --f16
```

## Architecture

```
model.vindex/           (flat binary files, mmap'd)
├── gate_vectors.bin    W_gate [layers × features × hidden] f16
├── up_weights.bin      W_up   [layers × features × hidden] f16
├── down_weights.bin    W_down [layers × hidden × features] f16
├── attn_weights.bin    Q/K/V/O + QK norms per layer, f16   ← full Gemma attention (GQA, RoPE, …)
├── norms.bin           4 RMSNorm gammas per layer + final, f16
├── embeddings.bin      [vocab × hidden] f16
├── tokenizer.json      (optional, for --prompt text I/O)
└── index.json          config, dimensions, layer info

Optional A-Vindex attention sidecar (same layout as avindex_attention.py extract):
├── attn_index.json           offsets / shapes for centroids + k_proj slices
├── attn_centroids.bin        MiniBatchKMeans centroids in projected K space (f32)
└── attn_k_proj_weights.bin   per-layer KV-head k_proj slices (f32)

These sidecar files are diagnostic only: they do not replace attn_weights.bin.
Build them in Rust (Lloyd k-means on projected vocab rows; centroids differ from Python sklearn) or with the optional Python `avindex_attention.py`:
  vindex-infer extract-attn-avindex --model-dir … --vindex …
After building them next to the vindex, run e.g.
  vindex-infer --vindex model.vindex -p "The capital of France is" --attn-vindex-probe
  vindex-infer ... --attn-vindex-scan --attn-scan-layers 13,14

         │
         ▼
   ┌─────────────┐
   │ vindex-infer │──→ CPU (rayon, any platform)
   │   (Rust)     │──→ Vulkan GPU (any vendor)
   └─────────────┘
         │
         ▼
   Paris (+21.24)
```

## Credits

- **[Gregor Koch](https://github.com/cronos3k/vindex-infer/tree/main)** — original **vindex-infer** Rust project and inference engine on GitHub (`cronos3k/vindex-infer`).
- **[LarQL](https://github.com/chrishayuk/larql)** by Chris Hayuk — the model decomposition engine that creates vindexes. All vindex format design, extraction pipeline, and LQL query language are his work.
- **[LarQL CUDA fork](https://github.com/cronos3k/larql)** — Linux/Windows + CUDA backend port.

## License

Apache-2.0
