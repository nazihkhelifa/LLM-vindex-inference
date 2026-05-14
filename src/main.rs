//! vindex-infer: Vendor-free LLM inference from decomposed weights.

mod attn_avindex;
mod extract_attn_avindex;
mod hf_model;
mod inference;
mod vindex;

use std::collections::HashSet;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "check-hf-model")]
struct CheckHfCli {
    /// Directory that should contain Hugging Face `config.json` and `*.safetensors`.
    #[arg(long)]
    model_dir: PathBuf,
}

#[derive(Parser)]
#[command(name = "download-hf-model")]
struct DownloadHfCli {
    /// Hugging Face model id, e.g. `google/gemma-3-4b-it`.
    #[arg(long)]
    hf_repo: String,
    /// Target directory (`hf download --local-dir`). If omitted, uses `./hf_checkout/<repo>/` under the current directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Read token for gated models (also read from env `HF_TOKEN` if unset).
    #[arg(long = "hf-token", env = "HF_TOKEN", hide_env_values = true)]
    hf_token: Option<String>,
}

#[derive(Copy, Clone, Default, Debug, ValueEnum)]
enum ForwardMode {
    /// Full transformer (attention + FFN each layer).
    #[default]
    Full,
    /// Ablation: skip attention sub-layer; FFN still runs (not HF-equivalent).
    FfnOnly,
}

#[derive(Parser)]
#[command(name = "vindex-infer")]
#[command(about = "Vendor-free LLM inference from decomposed weights. CPU or Vulkan GPU.")]
struct Args {
    /// Path to .vindex directory
    #[arg(short, long)]
    vindex: PathBuf,

    /// Optional path to `attn_weights.bin` when it is not inside `--vindex` (FFN + index in one dir, attention blob elsewhere).
    #[arg(long)]
    attn_weights: Option<PathBuf>,

    /// Input prompt (requires tokenizer.json in vindex dir, or --tokenizer)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Explicit token IDs (comma-separated). Overrides --prompt when set.
    #[arg(long)]
    token_ids: Option<String>,

    /// Hugging Face tokenizer.json path (default: <vindex>/tokenizer.json)
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Top-K predictions to show
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// GPU index for Vulkan backend (omit for CPU-only)
    #[arg(long)]
    gpu: Option<usize>,

    /// `full` = standard Gemma path; `ffn-only` = skip attention each layer (diagnostic).
    #[arg(long, value_enum, default_value_t = ForwardMode::Full)]
    forward: ForwardMode,

    /// After inference, run A-Vindex centroid probe (requires `attn_index.json`,
    /// `attn_centroids.bin`, `attn_k_proj_weights.bin` from `avindex_attention.py` extract).
    #[arg(long)]
    attn_vindex_probe: bool,

    /// After inference, scan all (layer, kv_head) centroid hits (diagnostic; can be verbose).
    #[arg(long)]
    attn_vindex_scan: bool,

    /// Layer for `--attn-vindex-probe` (-1 = middle of `layer_bands.knowledge` in index.json).
    #[arg(long, default_value_t = -1)]
    attn_probe_layer: i32,

    /// KV head index for `--attn-vindex-probe`.
    #[arg(long, default_value_t = 0)]
    attn_probe_kv_head: usize,

    /// Comma-separated layer ids for `--attn-vindex-scan` (empty = all layers).
    #[arg(long)]
    attn_scan_layers: Option<String>,

    /// Greedy multi-step generation: repeatedly append top-1 next token until EOS / stop token or `--max-new-tokens`.
    /// Requires `--prompt` and a tokenizer (same rules as text prompts). Incompatible with `--token-ids`.
    #[arg(long, conflicts_with = "token_ids")]
    conversation: bool,

    /// With `--conversation`: maximum number of new tokens to generate (safety cap).
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
}

fn tokenizer_path(args: &Args) -> PathBuf {
    args.tokenizer
        .clone()
        .unwrap_or_else(|| args.vindex.join("tokenizer.json"))
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    Tokenizer::from_file(path).map_err(|e| format!("{}: {}", path.display(), e).into())
}

/// Single-line, readable display for decoded subwords.
fn format_decoded(tokenizer: Option<&Tokenizer>, id: u32) -> String {
    let Some(tok) = tokenizer else {
        return format!("<{}>", id);
    };
    match tok.decode(&[id], false) {
        Ok(s) if !s.is_empty() => s
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t"),
        _ => format!("<{}>", id),
    }
}

/// Token ids that end greedy generation (resolved from the tokenizer when possible).
fn stop_token_ids_for_tokenizer(tok: &Tokenizer) -> HashSet<u32> {
    const NAMES: &[&str] = &[
        "<eos>",
        "</s>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end|>",
        "<end_of_turn>",
    ];
    let mut set = HashSet::new();
    for name in NAMES {
        if let Some(id) = tok.token_to_id(name) {
            set.insert(id);
        }
    }
    if set.is_empty() {
        set.insert(1);
    }
    set
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if argv.len() >= 2 && argv[1] == "check-hf-model" {
        argv.remove(1);
        let cli = CheckHfCli::parse_from(argv);
        let model_dir = hf_model::absolutize_model_dir(&cli.model_dir)?;
        if hf_model::hf_model_has_safetensors(&model_dir) {
            println!(
                "check-hf-model: OK — config.json and at least one *.safetensors under {}",
                model_dir.display()
            );
        } else {
            eprintln!("check-hf-model: INCOMPLETE\n{}", hf_model::diagnose(&model_dir));
            std::process::exit(1);
        }
        return Ok(());
    }
    if argv.len() >= 2 && argv[1] == "download-hf-model" {
        argv.remove(1);
        let cli = DownloadHfCli::parse_from(argv);
        let model_root = hf_model::resolve_model_dir_for_cli(cli.model_dir.as_deref(), Some(cli.hf_repo.as_str()))?;
        hf_model::ensure_hf_weights(&model_root, Some(cli.hf_repo.as_str()), cli.hf_token.as_deref())?;
        println!("download-hf-model: ready — {}", model_root.display());
        return Ok(());
    }
    if argv.len() >= 2 && argv[1] == "extract-attn-avindex" {
        argv.remove(1);
        let cli = extract_attn_avindex::ExtractCli::parse_from(argv);
        return extract_attn_avindex::run(cli);
    }

    let args = Args::parse();

    if args.conversation && args.prompt.is_none() {
        return Err("--conversation requires --prompt (and a tokenizer)".into());
    }

    println!("vindex-infer — Vendor-free LLM inference");
    println!();

    // Load vindex
    let vindex = vindex::Vindex::load_with_attn_weights(
        &args.vindex,
        args.attn_weights.as_deref(),
    )?;
    if (args.attn_vindex_probe || args.attn_vindex_scan)
        && !attn_avindex::sidecar_complete(&args.vindex)
    {
        return Err(
            "A-Vindex flags require attn_index.json, attn_centroids.bin, and attn_k_proj_weights.bin \
             next to the vindex. Build with: vindex-infer extract-attn-avindex --model-dir … --vindex …"
                .into(),
        );
    }
    println!(
        "Model: {} ({} layers, h={}, vocab={})",
        vindex.config.model,
        vindex.config.num_layers,
        vindex.config.hidden_size,
        vindex.config.vocab_size
    );

    let tok_path = tokenizer_path(&args);
    let tokenizer = if tok_path.is_file() {
        Some(load_tokenizer(&tok_path)?)
    } else if args.tokenizer.is_some() {
        return Err(format!("Tokenizer file not found: {}", tok_path.display()).into());
    } else {
        None
    };

    // Tokenize
    let mut token_ids: Vec<u32> = if let Some(ref ids) = args.token_ids {
        ids.split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid --token-ids: {}", e))?
    } else if let Some(ref prompt) = args.prompt {
        let tok = tokenizer.as_ref().ok_or_else(|| {
            format!(
                "Text prompts need a tokenizer. Add tokenizer.json under the vindex directory, or pass --tokenizer PATH.\n\
                 (Missing: {})",
                tok_path.display()
            )
        })?;
        // Match HF causal-LM style: encode prompt only; BOS is added below if absent.
        let enc = tok
            .encode(prompt.as_str(), false)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        enc.get_ids().to_vec()
    } else {
        // Default: "The capital of France is" (Gemma 3 token IDs)
        vec![818, 5279, 529, 7001, 563]
    };

    // Prepend BOS if missing (Gemma: 2)
    if token_ids.first() != Some(&2) {
        token_ids.insert(0, 2);
    }

    if let Some(ref tok) = tokenizer {
        // skip_special_tokens: hide BOS etc. for readable user-facing prompt line
        match tok.decode(&token_ids, true) {
            Ok(text) => println!("Prompt (decoded): {}", text.replace('\n', "\\n")),
            Err(_) => println!("Prompt: <decode error>"),
        }
    }
    if !args.conversation {
        println!("Token IDs: {:?}", token_ids);
    } else {
        println!("Token IDs (prompt, {} ids): {:?}", token_ids.len(), token_ids);
    }
    println!("Backend: {}", if args.gpu.is_some() { "Vulkan GPU" } else { "CPU (rayon)" });
    println!("Forward: {:?}", args.forward);
    if args.conversation {
        println!(
            "Mode: conversation (greedy top-1, max {} new tokens)",
            args.max_new_tokens
        );
    }
    println!();

    if args.conversation {
        let tok = tokenizer.as_ref().expect("conversation implies tokenizer loaded");
        let stop_ids = stop_token_ids_for_tokenizer(tok);
        let prompt_len = token_ids.len();
        println!("Greedy generation (stream):");
        print!("  ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let mut forward_ms = 0.0f64;
        let mut layer_ms = 0.0f64;
        let mut logits_ms = 0.0f64;
        let mut steps = 0usize;

        for _ in 0..args.max_new_tokens {
            let result = match args.forward {
                ForwardMode::Full => inference::infer(&vindex, &token_ids, 1),
                ForwardMode::FfnOnly => inference::infer_ffn_only(&vindex, &token_ids, 1),
            };
            forward_ms += result.total_ms;
            layer_ms += result.layer_ms;
            logits_ms += result.logits_ms;

            let (next_id, score) = result.top_k.first().copied().ok_or("empty top_k")?;
            if stop_ids.contains(&next_id) {
                println!();
                println!(
                    "Stopped on end token id {} (score {:+.4}); not appended.",
                    next_id, score
                );
                break;
            }
            token_ids.push(next_id);
            steps += 1;
            let piece = tok.decode(&[next_id], false).unwrap_or_default();
            print!("{}", piece);
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        if steps == args.max_new_tokens {
            println!();
            println!(
                "Stopped: reached --max-new-tokens ({}) without an end token.",
                args.max_new_tokens
            );
        } else if steps > 0 {
            println!();
        }

        println!();
        println!("Full continuation (decoded, skip_special_tokens):");
        let tail = &token_ids[prompt_len..];
        match tok.decode(tail, true) {
            Ok(s) => println!("{}", s),
            Err(_) => println!("<decode error>"),
        }
        println!();
        println!(
            "Conversation: {} new tokens | forward sum {:.1}ms (layers {:.0}ms, logits {:.0}ms)",
            steps, forward_ms, layer_ms, logits_ms
        );
    } else {
        // Run inference (single forward)
        let result = match args.forward {
            ForwardMode::Full => inference::infer(&vindex, &token_ids, args.top_k),
            ForwardMode::FfnOnly => inference::infer_ffn_only(&vindex, &token_ids, args.top_k),
        };

        // Display results
        println!("Predictions:");
        for (i, (tid, score)) in result.top_k.iter().enumerate() {
            let piece = format_decoded(tokenizer.as_ref(), *tid);
            println!(
                "  {:2}. {:<24}  id {:6}  score {:+.4}",
                i + 1,
                piece,
                tid,
                score
            );
        }

        println!();
        println!(
            "Time: {:.1}ms total ({:.0}ms layers, {:.0}ms logits)",
            result.total_ms, result.layer_ms, result.logits_ms
        );
    }

    if args.attn_vindex_probe || args.attn_vindex_scan {
        let reader = attn_avindex::AttnAvindexReader::open(&args.vindex)?;
        let last_tid = *token_ids.last().expect("non-empty tokens") as usize;
        let e = vindex.embedding(last_tid);

        if args.attn_vindex_probe {
            let main_raw = std::fs::read_to_string(args.vindex.join("index.json"))?;
            let main_v: serde_json::Value = serde_json::from_str(&main_raw)?;
            let layer = if args.attn_probe_layer < 0 {
                attn_avindex::default_probe_layer(&main_v)
            } else {
                args.attn_probe_layer as usize
            };
            if layer >= reader.num_layers() {
                return Err(format!(
                    "--attn-probe-layer {} is out of range (num_layers {})",
                    layer,
                    reader.num_layers()
                )
                .into());
            }
            if args.attn_probe_kv_head >= reader.n_kv(layer) {
                return Err(format!(
                    "--attn-probe-kv-head {} is out of range for layer {} (n_kv={})",
                    args.attn_probe_kv_head,
                    layer,
                    reader.n_kv(layer)
                )
                .into());
            }
            let out = attn_avindex::probe_sidecar(&reader, &e, layer, args.attn_probe_kv_head)?;
            println!();
            println!("A-Vindex probe (last-token embedding × stored k_proj → centroid; diagnostic only):");
            println!(
                "  layer={} kv_head={}  centroid_id={}  cosine={:+.6}",
                out.layer, out.kv_head, out.centroid_id, out.cosine
            );
        }

        if args.attn_vindex_scan {
            let layers_only: Option<Vec<usize>> = args.attn_scan_layers.as_ref().and_then(|s| {
                let v: Vec<usize> = s
                    .split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if v.is_empty() {
                    None
                } else {
                    Some(v)
                }
            });
            let lo = layers_only.as_deref();
            let steps = attn_avindex::scan_sidecar(&reader, &e, lo)?;
            println!();
            println!(
                "A-Vindex scan: {} (layer, kv_head) steps (showing first 24)",
                steps.len()
            );
            for s in steps.iter().take(24) {
                println!(
                    "  layer {:2} kv {:1}  centroid {:4}  cosine {:+.5}",
                    s.layer, s.kv_head, s.centroid_id, s.cosine
                );
            }
            if steps.len() > 24 {
                println!("  …");
            }
        }
    }

    Ok(())
}
