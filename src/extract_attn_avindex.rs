//! Build A-Vindex attention sidecar from HF `config.json` + `*.safetensors`.
//! Lloyd k-means with random init (distinct seeds per head). File layout matches `attn_avindex`.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use crate::vindex::f16_to_f32;

#[derive(Clone, Copy, Default, Debug, ValueEnum)]
pub enum OutputDtype {
    /// IEEE f16 on disk (smaller sidecar; default).
    #[default]
    F16,
    /// IEEE f32 on disk (matches legacy Python extract).
    F32,
}

#[derive(Parser, Debug)]
#[command(name = "extract-attn-avindex")]
pub struct ExtractCli {
    /// Directory with Hugging Face `config.json` and `*.safetensors`. If omitted while `--hf-repo` is set, weights are stored under `./hf_checkout/<repo>/` (from the current working directory).
    #[arg(long)]
    pub model_dir: Option<std::path::PathBuf>,

    /// If `model_dir` lacks `config.json` or `*.safetensors`, download this repo via `hf download` (needs `hf` on PATH).
    #[arg(long)]
    pub hf_repo: Option<String>,

    /// Hugging Face token for gated models (passed to `hf` as `HF_TOKEN`). Same as env `HF_TOKEN`.
    #[arg(long = "hf-token", env = "HF_TOKEN", hide_env_values = true)]
    pub hf_token: Option<String>,

    #[arg(short, long)]
    pub vindex: std::path::PathBuf,

    #[arg(long, default_value_t = 256)]
    pub centroids: usize,

    #[arg(long, default_value_t = 8000)]
    pub vocab_sample: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 25)]
    pub kmeans_iters: usize,

    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Binary layout for `attn_centroids.bin` and `attn_k_proj_weights.bin` (recorded in `attn_index.json` `dtype`).
    #[arg(long, value_enum, default_value_t = OutputDtype::F16)]
    pub output_dtype: OutputDtype,
}

/// HF configs like `Gemma3ForConditionalGeneration` nest LM fields under `text_config`.
fn hf_lm_subconfig(cfg: &serde_json::Value) -> &serde_json::Value {
    match cfg.get("text_config") {
        Some(tc) if !tc.is_null() && tc.is_object() => {
            if tc.get("num_hidden_layers").is_some() || tc.get("hidden_size").is_some() {
                tc
            } else {
                cfg
            }
        }
        _ => cfg,
    }
}

fn json_u64(o: &serde_json::Value, key: &str) -> Option<u64> {
    o.get(key).and_then(|v| v.as_u64())
}

/// `num_hidden_layers` / `hidden_size` from root or `text_config`.
fn pick_num_layers_hidden(cfg: &serde_json::Value) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let lm = hf_lm_subconfig(cfg);
    let num_layers = json_u64(lm, "num_hidden_layers")
        .or_else(|| json_u64(cfg, "num_hidden_layers"))
        .ok_or(
            "config: missing num_hidden_layers (expected at config root or under text_config for Gemma3 multimodal)",
        )? as usize;
    let hidden = json_u64(lm, "hidden_size")
        .or_else(|| json_u64(cfg, "hidden_size"))
        .ok_or("config: missing hidden_size (root or text_config)")? as usize;
    Ok((num_layers, hidden))
}

/// Optional hints from LarQL `index.json` next to the vindex (vocab + head counts for Gemma3 when HF config is minimal).
fn read_vindex_index_hints(vindex_dir: &Path) -> Result<Option<(Option<u64>, Option<u64>, Option<u64>)>, Box<dyn std::error::Error>> {
    let p = vindex_dir.join("index.json");
    if !p.is_file() {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&p)?)?;
    let vocab = v.get("vocab_size").and_then(|x| x.as_u64());
    let (nq, nkv) = match v.get("model_config") {
        Some(mc) => (
            mc.get("num_q_heads").and_then(|x| x.as_u64()),
            mc.get("num_kv_heads").and_then(|x| x.as_u64()),
        ),
        None => (None, None),
    };
    Ok(Some((vocab, nq, nkv)))
}

fn pick_vocab_and_heads(
    cfg: &serde_json::Value,
    vindex_dir: &Path,
) -> Result<(Option<usize>, Option<usize>, Option<usize>), Box<dyn std::error::Error>> {
    let lm = hf_lm_subconfig(cfg);
    let mut vocab = json_u64(lm, "vocab_size")
        .or_else(|| json_u64(cfg, "vocab_size"))
        .map(|x| x as usize);
    let mut n_heads = json_u64(lm, "num_attention_heads")
        .or_else(|| json_u64(lm, "num_heads"))
        .or_else(|| json_u64(cfg, "num_attention_heads"))
        .or_else(|| json_u64(cfg, "num_heads"))
        .map(|x| x as usize);
    let mut n_kv = json_u64(lm, "num_key_value_heads")
        .or_else(|| json_u64(cfg, "num_key_value_heads"))
        .map(|x| x as usize);

    if let Some((v2, q2, kv2)) = read_vindex_index_hints(vindex_dir)? {
        vocab = vocab.or(v2.map(|x| x as usize));
        n_heads = n_heads.or(q2.map(|x| x as usize));
        n_kv = n_kv.or(kv2.map(|x| x as usize));
    }

    Ok((vocab, n_heads, n_kv))
}

fn infer_vocab_from_embed_table(
    tensors: &HashMap<String, Vec<f32>>,
    prefix: &str,
    hidden: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let cands = [
        format!("{}embed_tokens.weight", prefix),
        "model.embed_tokens.weight".to_string(),
        "embed_tokens.weight".to_string(),
    ];
    for c in &cands {
        if let Some(t) = tensors.get(c) {
            if t.len() % hidden != 0 {
                continue;
            }
            let v = t.len() / hidden;
            if v == 0 {
                continue;
            }
            log::info!(
                "[extract-attn-avindex] inferred vocab_size={} from tensor `{}` (len={}, hidden={})",
                v,
                c,
                t.len(),
                hidden
            );
            return Ok(v);
        }
    }
    Err(
        "Could not infer vocab_size: missing in HF config and no embed_tokens.weight tensor matched. \
         Add vocab_size to HF config or use a vindex directory whose index.json lists vocab_size."
            .into(),
    )
}

pub fn run(cli: ExtractCli) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("[extract-attn-avindex] start");
    fs::create_dir_all(&cli.vindex)?;
    let out_dir = cli.vindex.canonicalize()?;
    log::info!(
        "[extract-attn-avindex] vindex output dir: {}",
        out_dir.display()
    );

    let model_root = crate::hf_model::resolve_model_dir_for_cli(
        cli.model_dir.as_deref(),
        cli.hf_repo.as_deref(),
    )?;
    log::info!(
        "[extract-attn-avindex] resolved model dir: {}",
        model_root.display()
    );
    log::info!("[extract-attn-avindex] ensure HF weights (download if needed)…");
    crate::hf_model::ensure_hf_weights(&model_root, cli.hf_repo.as_deref(), cli.hf_token.as_deref())?;
    let model_dir = model_root.canonicalize()?;
    log::info!(
        "[extract-attn-avindex] model dir (canonical): {}",
        model_dir.display()
    );

    if !cli.force && crate::attn_avindex::sidecar_complete(&out_dir) {
        log::info!(
            "[extract-attn-avindex] SKIP — sidecar already present under {} (use --force)",
            out_dir.display()
        );
        println!(
            "[extract-attn-avindex] SKIP — sidecar already present under {} (use --force)",
            out_dir.display()
        );
        return Ok(());
    }

    log::info!("[extract-attn-avindex] read config.json …");
    let cfg_raw = fs::read_to_string(model_dir.join("config.json"))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_raw)?;
    let (num_layers, hidden) = pick_num_layers_hidden(&cfg)?;
    let (vocab_opt, n_heads_opt, n_kv_opt) = pick_vocab_and_heads(&cfg, &out_dir)?;

    let n_heads = n_heads_opt.ok_or(
        "config: missing num_attention_heads / num_heads (HF root, text_config, or vindex index.json model_config)",
    )?;
    let n_kv = n_kv_opt.unwrap_or(n_heads);

    log::info!("[extract-attn-avindex] load all *.safetensors under model dir …");
    let tensors = load_all_f32_tensors(&model_dir)?;
    log::info!(
        "[extract-attn-avindex] loaded {} unique tensor names from safetensors",
        tensors.len()
    );
    let keys: HashSet<String> = tensors.keys().cloned().collect();
    log::info!("[extract-attn-avindex] detect checkpoint key prefix …");
    let prefix = detect_prefix(&keys)?;
    log::info!("[extract-attn-avindex] prefix: {:?}", prefix);

    let vocab = match vocab_opt {
        Some(v) => v,
        None => infer_vocab_from_embed_table(&tensors, &prefix, hidden)?,
    };

    println!(
        "[extract-attn-avindex] model_dir={}\n                        vindex={}\n                        layers={} hidden={} vocab={} q_heads={} kv_heads={}\n                        output_dtype={:?}",
        model_dir.display(),
        out_dir.display(),
        num_layers,
        hidden,
        vocab,
        n_heads,
        n_kv,
        cli.output_dtype
    );
    log::info!(
        "[extract-attn-avindex] config: layers={} hidden={} vocab={} q_heads={} kv_heads={} output_dtype={:?}",
        num_layers,
        hidden,
        vocab,
        n_heads,
        n_kv,
        cli.output_dtype
    );

    log::info!("[extract-attn-avindex] locate embedding table …");
    let emb = find_embed(&tensors, &prefix, vocab, hidden)?;
    log::info!(
        "[extract-attn-avindex] embedding table: {} floats (vocab×hidden)",
        emb.len()
    );
    let n_sample = cli.vocab_sample.min(vocab);
    log::info!(
        "[extract-attn-avindex] vocab subsample: {} / {} token rows for k-means",
        n_sample,
        vocab
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(cli.seed);
    let mut sample_idx: Vec<usize> = (0..vocab).collect();
    sample_idx.shuffle(&mut rng);
    sample_idx.truncate(n_sample);
    sample_idx.sort_unstable();

    let mut emb_s = vec![0f32; n_sample * hidden];
    for (si, &vid) in sample_idx.iter().enumerate() {
        emb_s[si * hidden..(si + 1) * hidden].copy_from_slice(&emb[vid * hidden..(vid + 1) * hidden]);
    }

    let mut layers_arr: Vec<serde_json::Value> = Vec::with_capacity(num_layers);
    let mut kproj_arr: Vec<serde_json::Value> = Vec::with_capacity(num_layers);

    let centroid_path = out_dir.join("attn_centroids.bin");
    let kproj_path = out_dir.join("attn_k_proj_weights.bin");
    log::info!(
        "[extract-attn-avindex] create sidecar binaries: {} , {}",
        centroid_path.display(),
        kproj_path.display()
    );
    let mut f_cent = File::create(&centroid_path)?;
    let mut f_k = File::create(&kproj_path)?;
    let mut k_file_off: usize = 0;
    let mut c_file_off: usize = 0;

    for layer in 0..num_layers {
        log::info!(
            "[extract-attn-avindex] layer {}/{}: load k_proj …",
            layer + 1,
            num_layers
        );
        let wk = find_k_proj(&tensors, &prefix, layer, hidden)?;
        let (wk_flat, rows, cols) = wk;
        let (wk_kv_flat, head_dim) = reshape_k_proj(&wk_flat, rows, cols, n_kv, hidden)?;
        let mut layer_cent_kv: Vec<serde_json::Value> = Vec::with_capacity(n_kv);
        let mut layer_kproj_kv: Vec<serde_json::Value> = Vec::with_capacity(n_kv);
        for kv_h in 0..n_kv {
            log::debug!(
                "[extract-attn-avindex] layer {} kv_head {}/{} (head_dim={})",
                layer,
                kv_h + 1,
                n_kv,
                head_dim
            );
            let base = kv_h * head_dim * hidden;
            let head_k = &wk_kv_flat[base..base + head_dim * hidden];
            let k_off = k_file_off;
            write_storage(&mut f_k, head_k, cli.output_dtype)?;
            let k_bytes = storage_bytes(head_k.len(), cli.output_dtype);
            k_file_off += k_bytes;
            layer_kproj_kv.push(serde_json::json!({
                "kv_head": kv_h,
                "offset": k_off,
                "size_bytes": k_bytes,
                "shape": [head_dim, hidden],
            }));

            let mut projected = vec![0f32; n_sample * head_dim];
            for si in 0..n_sample {
                for d in 0..head_dim {
                    let mut s = 0f32;
                    for h in 0..hidden {
                        s += emb_s[si * hidden + h] * head_k[d * hidden + h];
                    }
                    projected[si * head_dim + d] = s;
                }
            }
            let k_use = cli.centroids.min(n_sample).max(1);
            let head_seed = cli.seed.wrapping_add((layer as u64) << 16 | kv_h as u64);
            log::debug!(
                "[extract-attn-avindex] layer {} kv_head {}: k-means Lloyd (k={}, max_iter={}) …",
                layer,
                kv_h,
                k_use,
                cli.kmeans_iters
            );
            let centroids = kmeans_lloyd(
                &projected,
                n_sample,
                head_dim,
                k_use,
                cli.kmeans_iters,
                head_seed,
            );
            let c_off = c_file_off;
            write_storage(&mut f_cent, &centroids, cli.output_dtype)?;
            let c_bytes = storage_bytes(centroids.len(), cli.output_dtype);
            c_file_off += c_bytes;
            layer_cent_kv.push(serde_json::json!({
                "kv_head": kv_h,
                "offset": c_off,
                "size_bytes": c_bytes,
                "shape": [k_use, head_dim],
            }));
        }
        layers_arr.push(serde_json::json!({
            "layer": layer,
            "head_dim_kv": head_dim,
            "kv_heads": layer_cent_kv,
        }));
        kproj_arr.push(serde_json::json!({
            "layer": layer,
            "head_dim_kv": head_dim,
            "kv_heads": layer_kproj_kv,
        }));
        println!(
            "[extract-attn-avindex] layer {}/{}  head_dim_kv={}",
            layer + 1,
            num_layers,
            head_dim
        );
        log::info!(
            "[extract-attn-avindex] layer {}/{} done (head_dim_kv={}, {} kv heads)",
            layer + 1,
            num_layers,
            head_dim,
            n_kv
        );
    }

    log::info!("[extract-attn-avindex] write attn_index.json …");
    let head_dim_kv = layers_arr
        .first()
        .and_then(|v| v.get("head_dim_kv"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let dtype_str = match cli.output_dtype {
        OutputDtype::F16 => "f16",
        OutputDtype::F32 => "f32",
    };
    let attn_index = serde_json::json!({
        "avindex_version": 3,
        "kind": "k_proj_vocab_projection_centroids",
        "num_layers": num_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "hidden_size": hidden,
        "vocab_sample_size": n_sample,
        "num_centroids": cli.centroids,
        "dtype": dtype_str,
        "extractor": "vindex-infer-rust-lloyd-kmeans",
        "head_dim_kv": head_dim_kv,
        "layers": layers_arr,
        "k_proj_layers": kproj_arr,
    });

    fs::write(
        out_dir.join("attn_index.json"),
        serde_json::to_string_pretty(&attn_index)?,
    )?;

    println!(
        "[extract-attn-avindex] wrote {}  {}  attn_index.json",
        centroid_path.display(),
        kproj_path.display()
    );
    log::info!(
        "[extract-attn-avindex] finished: {} , {} , attn_index.json",
        centroid_path.display(),
        kproj_path.display()
    );
    Ok(())
}

fn storage_bytes(num_elems: usize, dt: OutputDtype) -> usize {
    num_elems
        * match dt {
            OutputDtype::F16 => 2,
            OutputDtype::F32 => 4,
        }
}

fn write_storage(w: &mut File, data: &[f32], dt: OutputDtype) -> std::io::Result<()> {
    match dt {
        OutputDtype::F32 => write_f32_le(w, data),
        OutputDtype::F16 => {
            for &x in data {
                let bits = f16::from_f32(x).to_bits();
                w.write_all(&bits.to_le_bytes())?;
            }
            Ok(())
        }
    }
}

fn write_f32_le(w: &mut File, data: &[f32]) -> std::io::Result<()> {
    for &x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn load_all_f32_tensors(model_dir: &Path) -> Result<HashMap<String, Vec<f32>>, Box<dyn std::error::Error>> {
    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    let mut shard_paths: Vec<std::path::PathBuf> = Vec::new();
    for ent in fs::read_dir(model_dir)? {
        let p = ent?.path();
        if p.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        shard_paths.push(p);
    }
    shard_paths.sort();
    log::info!(
        "[extract-attn-avindex] found {} safetensor shard file(s)",
        shard_paths.len()
    );
    for p in &shard_paths {
        let shard_name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        log::info!(
            "[extract-attn-avindex]   read shard `{}` …",
            shard_name
        );
        let bytes = fs::read(p)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let n_tensors = st.tensors().len();
        for (tname, view) in st.tensors() {
            let v = tensor_view_to_f32(&view)?;
            out.insert(tname.to_string(), v);
        }
        log::info!(
            "[extract-attn-avindex]   shard `{}`: {} tensors ({} total keys so far)",
            shard_name,
            n_tensors,
            out.len()
        );
    }
    if out.is_empty() {
        return Err(format!("no .safetensors tensors under {}", model_dir.display()).into());
    }
    Ok(out)
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                return Err("bad f32 tensor".into());
            }
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                return Err("bad f16 tensor".into());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                return Err("bad bf16 tensor".into());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        d => Err(format!("unsupported dtype {:?} (need F32, F16, or BF16)", d).into()),
    }
}

fn detect_prefix(keys: &HashSet<String>) -> Result<String, Box<dyn std::error::Error>> {
    let needle = "layers.0.self_attn.k_proj.weight";
    for k in keys {
        if let Some(pos) = k.find(needle) {
            return Ok(k[..pos].to_string());
        }
    }
    Err("could not find *layers.0.self_attn.k_proj.weight* in safetensors keys".into())
}

fn find_embed(
    m: &HashMap<String, Vec<f32>>,
    prefix: &str,
    vocab: usize,
    hidden: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let cands = [
        format!("{}embed_tokens.weight", prefix),
        "model.embed_tokens.weight".to_string(),
        "embed_tokens.weight".to_string(),
    ];
    for c in &cands {
        if let Some(t) = m.get(c) {
            return verify_embed(t, vocab, hidden);
        }
    }
    Err(format!("embed not found; tried {:?}", cands).into())
}

fn verify_embed(t: &[f32], vocab: usize, hidden: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if t.len() == vocab * hidden {
        return Ok(t.to_vec());
    }
    if t.len() == hidden * vocab {
        let mut out = vec![0f32; vocab * hidden];
        for v in 0..vocab {
            for h in 0..hidden {
                out[v * hidden + h] = t[h * vocab + v];
            }
        }
        return Ok(out);
    }
    Err(format!("bad embed len {} for vocab×hidden={}", t.len(), vocab * hidden).into())
}

fn find_k_proj(
    m: &HashMap<String, Vec<f32>>,
    prefix: &str,
    layer: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let cands = [
        format!("{}layers.{layer}.self_attn.k_proj.weight", prefix),
        format!("model.layers.{layer}.self_attn.k_proj.weight"),
    ];
    for c in &cands {
        if let Some(t) = m.get(c) {
            if t.len() % hidden != 0 {
                continue;
            }
            let out_f = t.len() / hidden;
            return Ok((t.clone(), out_f, hidden));
        }
    }
    Err(format!("k_proj not found for layer {}", layer).into())
}

fn reshape_k_proj(
    w: &[f32],
    rows: usize,
    cols: usize,
    n_kv: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize), Box<dyn std::error::Error>> {
    let (flat, out_dim, in_dim) = if cols == hidden && rows % n_kv == 0 {
        (w.to_vec(), rows, cols)
    } else if rows == hidden && cols % n_kv == 0 {
        let mut t = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = w[i * cols + j];
            }
        }
        (t, cols, rows)
    } else {
        return Err(format!("unexpected k_proj {}×{} hidden={} n_kv={}", rows, cols, hidden, n_kv).into());
    };
    if in_dim != hidden {
        return Err(format!("k_proj in_dim {} != hidden {}", in_dim, hidden).into());
    }
    if out_dim % n_kv != 0 {
        return Err("k_proj out_dim not divisible by n_kv".into());
    }
    let head_dim = out_dim / n_kv;
    Ok((flat, head_dim))
}

fn kmeans_lloyd(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let k = k.min(n).max(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    let mut centroids = vec![0f32; k * d];
    for j in 0..k {
        let pick = idx[j % n];
        copy_point(data, d, pick, &mut centroids, j);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = 0usize;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for j in 0..k {
                let dd = dist2_row(data, d, i, &centroids, j);
                if dd < best_d {
                    best_d = dd;
                    best = j;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed += 1;
            }
        }
        centroids.fill(0.0);
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            for t in 0..d {
                centroids[c * d + t] += data[i * d + t];
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                let steal = (j.wrapping_mul(7919).wrapping_add(17)) % n;
                copy_point(data, d, steal, &mut centroids, j);
                counts[j] = 1;
            } else {
                let inv = 1.0f32 / counts[j] as f32;
                for t in 0..d {
                    centroids[j * d + t] *= inv;
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    centroids
}

fn copy_point(data: &[f32], d: usize, pi: usize, centroids: &mut [f32], ci: usize) {
    centroids[ci * d..(ci + 1) * d].copy_from_slice(&data[pi * d..(pi + 1) * d]);
}

fn dist2_row(data: &[f32], d: usize, pi: usize, centroids: &[f32], ci: usize) -> f32 {
    let mut s = 0f64;
    for t in 0..d {
        let u = data[pi * d + t] - centroids[ci * d + t];
        s += (u as f64) * (u as f64);
    }
    s as f32
}
