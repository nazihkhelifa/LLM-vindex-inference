//! Optional **A-Vindex attention sidecar** — same on-disk layout as `avindex_attention.py`
//! (`attn_centroids.bin`, `attn_k_proj_weights.bin`, `attn_index.json`).
//!
//! This does **not** replace full attention (`attn_weights.bin` + `inference::infer`). It is a
//! diagnostic projection: last-token embedding row × stored `k_proj` slice → nearest
//! MiniBatchKMeans centroid (cosine), matching the Python extract/probe space.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

fn is_f16_storage(meta: &Value) -> bool {
    let s = meta
        .get("dtype")
        .and_then(|x| x.as_str())
        .unwrap_or("f32");
    matches!(
        s.to_ascii_lowercase().as_str(),
        "f16" | "float16" | "fp16"
    )
}

/// Files written by `avindex_attention.py` / `extract_attention_avindex`.
pub fn sidecar_complete(vdir: &Path) -> bool {
    for name in ["attn_index.json", "attn_centroids.bin", "attn_k_proj_weights.bin"] {
        let p = vdir.join(name);
        if !p.is_file() || p.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug)]
struct HeadBlob {
    offset: usize,
    size_bytes: usize,
    shape: [usize; 2],
}

#[derive(Clone, Debug)]
struct LayerBlobs {
    head_dim_kv: usize,
    kv_heads: Vec<HeadBlob>,
}

#[derive(Debug)]
pub struct AttnAvindexReader {
    centroid_mmap: Mmap,
    kproj_mmap: Mmap,
    centroid_layers: Vec<LayerBlobs>,
    kproj_layers: Vec<LayerBlobs>,
    /// Element width in `attn_centroids.bin` / `attn_k_proj_weights.bin` (from `attn_index.json` `dtype`).
    blob_f16: bool,
}

impl AttnAvindexReader {
    pub fn open(vdir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let meta_path = vdir.join("attn_index.json");
        let raw = std::fs::read_to_string(&meta_path)?;
        let v: Value = serde_json::from_str(&raw)?;
        let blob_f16 = is_f16_storage(&v);
        let centroid_layers = parse_layers_field(v.get("layers"), "layers")?;
        let kproj_layers = if let Some(kp) = v.get("k_proj_layers") {
            parse_layers_field(Some(kp), "k_proj_layers")?
        } else {
            return Err("attn_index.json missing 'k_proj_layers' (re-run avindex_attention extract)".into());
        };

        let centroid_mmap = unsafe { Mmap::map(&File::open(vdir.join("attn_centroids.bin"))?)? };
        let kproj_mmap = unsafe { Mmap::map(&File::open(vdir.join("attn_k_proj_weights.bin"))?)? };

        Ok(Self {
            centroid_mmap,
            kproj_mmap,
            centroid_layers,
            kproj_layers,
            blob_f16,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.centroid_layers.len()
    }

    pub fn n_kv(&self, layer: usize) -> usize {
        self.centroid_layers[layer].kv_heads.len()
    }

    /// `k_proj` slice `[head_dim_kv, hidden]` as contiguous row-major `f32`.
    pub fn k_slice(&self, layer: usize, kv_head: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let lb = &self.kproj_layers[layer];
        let hb = &lb.kv_heads[kv_head];
        self.read_weight_blob(&self.kproj_mmap, hb, lb.head_dim_kv, self.blob_f16)
    }

    /// Centroid matrix `[num_centroids, head_dim_kv]`.
    pub fn centroids(&self, layer: usize, kv_head: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let lb = &self.centroid_layers[layer];
        let hb = &lb.kv_heads[kv_head];
        self.read_weight_blob(&self.centroid_mmap, hb, lb.head_dim_kv, self.blob_f16)
    }

    fn read_weight_blob(
        &self,
        mmap: &Mmap,
        hb: &HeadBlob,
        head_dim_fallback: usize,
        f16: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let start = hb.offset;
        let end = start + hb.size_bytes;
        if end > mmap.len() {
            return Err(format!(
                "blob out of range: offset {} size {} file_len {}",
                start,
                hb.size_bytes,
                mmap.len()
            )
            .into());
        }
        let raw = &mmap[start..end];
        let elem = if f16 { 2usize } else { 4usize };
        let (rows, cols) = if hb.shape[0] > 0 && hb.shape[1] > 0 {
            (hb.shape[0], hb.shape[1])
        } else {
            let d = head_dim_fallback;
            let n = raw.len() / (elem * d.max(1));
            (n, d)
        };
        let need = rows * cols * elem;
        if raw.len() < need {
            return Err(format!(
                "blob size {} < expected {} elements (dtype {})",
                raw.len(),
                rows * cols,
                if f16 { "f16" } else { "f32" }
            )
            .into());
        }
        let mut out = Vec::with_capacity(rows * cols);
        if f16 {
            for chunk in raw[..need].chunks_exact(2) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(crate::vindex::f16_to_f32(u));
            }
        } else {
            for chunk in raw[..need].chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Ok(out)
    }

    pub fn nearest_cosine(
        centroids_rowmajor: &[f32],
        head_dim: usize,
        q: &[f32],
    ) -> Result<(usize, f32), Box<dyn std::error::Error>> {
        if q.len() != head_dim {
            return Err(format!("q dim {} != head_dim {}", q.len(), head_dim).into());
        }
        let n = centroids_rowmajor.len() / head_dim;
        if n * head_dim != centroids_rowmajor.len() {
            return Err("centroid buffer length not divisible by head_dim".into());
        }
        let qn: f32 = q.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        let qscale = 1.0f32 / (qn + 1e-8);
        let mut best_j = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for j in 0..n {
            let row = &centroids_rowmajor[j * head_dim..(j + 1) * head_dim];
            let cn: f32 = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
            let cscale = 1.0f32 / (cn + 1e-8);
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += row[d] * cscale * q[d] * qscale;
            }
            if dot > best_s {
                best_s = dot;
                best_j = j;
            }
        }
        Ok((best_j, best_s))
    }

    /// Same math as Python `probe_last_token_k_space`: `q = e @ K^T` then nearest centroid.
    pub fn probe_embedding(
        &self,
        e_hidden: &[f32],
        layer: usize,
        kv_head: usize,
    ) -> Result<(usize, f32), Box<dyn std::error::Error>> {
        let hidden = e_hidden.len();
        let k = self.k_slice(layer, kv_head)?;
        let hd = self.centroid_layers[layer].head_dim_kv;
        let w = k.len() / hidden;
        if w * hidden != k.len() {
            return Err("k_proj length mismatch".into());
        }
        // k row-major [head_dim, hidden]: q[d] = sum_i e[i] * k[d*hidden + i]
        let mut q = vec![0f32; w];
        for d in 0..w {
            let row = &k[d * hidden..(d + 1) * hidden];
            q[d] = e_hidden.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
        }
        let c = self.centroids(layer, kv_head)?;
        Self::nearest_cosine(&c, hd, &q)
    }
}

fn parse_layers_field(v: Option<&Value>, label: &str) -> Result<Vec<LayerBlobs>, Box<dyn std::error::Error>> {
    let v = v.ok_or_else(|| format!("attn_index.json missing '{}'", label))?;
    let entries: Vec<Value> = match v {
        Value::Array(a) => a.clone(),
        Value::Object(o) => {
            let mut keys: Vec<i32> = o
                .keys()
                .filter_map(|k| k.parse::<i32>().ok())
                .collect();
            keys.sort_unstable();
            keys.into_iter().filter_map(|k| o.get(&k.to_string()).cloned()).collect()
        }
        _ => return Err(format!("'{}' must be array or object", label).into()),
    };
    let mut out = Vec::with_capacity(entries.len());
    for (i, ent) in entries.into_iter().enumerate() {
        let obj = ent.as_object().ok_or("layer entry must be object")?;
        let layer = obj
            .get("layer")
            .and_then(|x| x.as_u64())
            .map(|u| u as usize)
            .unwrap_or(i);
        let head_dim_kv = obj
            .get("head_dim_kv")
            .and_then(|x| x.as_u64())
            .map(|u| u as usize)
            .or_else(|| infer_head_dim_kv(obj))
            .ok_or_else(|| format!("layer {}: missing head_dim_kv", layer))?;
        let kv_arr = obj
            .get("kv_heads")
            .or_else(|| obj.get("heads"))
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("layer {}: missing kv_heads", layer))?;
        let mut kv_heads = Vec::with_capacity(kv_arr.len());
        for (hi, hv) in kv_arr.iter().enumerate() {
            let h = hv.as_object().ok_or("kv_head must be object")?;
            let offset = h
                .get("offset")
                .and_then(|x| x.as_u64())
                .ok_or("kv_head.offset")? as usize;
            let size_bytes = h
                .get("size_bytes")
                .and_then(|x| x.as_u64())
                .ok_or("kv_head.size_bytes")? as usize;
            let shape = h
                .get("shape")
                .and_then(|x| x.as_array())
                .and_then(|a| {
                    if a.len() == 2 {
                        Some([a[0].as_u64()? as usize, a[1].as_u64()? as usize])
                    } else {
                        None
                    }
                })
                .unwrap_or([0, 0]);
            kv_heads.push(HeadBlob {
                offset,
                size_bytes,
                shape,
            });
            let _ = hi;
        }
        out.push(LayerBlobs {
            head_dim_kv,
            kv_heads,
        });
    }
    Ok(out)
}

fn infer_head_dim_kv(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    let kv = obj.get("kv_heads")?.as_array()?;
    let h0 = kv.first()?.as_object()?;
    let sh = h0.get("shape")?.as_array()?;
    if sh.len() == 2 {
        Some(sh[1].as_u64()? as usize)
    } else {
        None
    }
}

/// Pick a default layer from `index.json` `layer_bands.knowledge` mid, else `nl / 2`.
pub fn default_probe_layer(main_index: &serde_json::Value) -> usize {
    if let Some(bands) = main_index.get("layer_bands").and_then(|b| b.as_object()) {
        if let Some(kn) = bands.get("knowledge").and_then(|k| k.as_array()) {
            if kn.len() == 2 {
                let a = kn[0].as_u64().unwrap_or(0) as usize;
                let b = kn[1].as_u64().unwrap_or(kn[0].as_u64().unwrap_or(0)) as usize;
                return (a + b) / 2;
            }
        }
    }
    let nl = main_index
        .get("num_layers")
        .and_then(|x| x.as_u64())
        .unwrap_or(1) as usize;
    nl.saturating_sub(1) / 2
}

pub struct AvindexProbeOut {
    pub layer: usize,
    pub kv_head: usize,
    pub centroid_id: usize,
    pub cosine: f32,
}

pub fn probe_sidecar(
    reader: &AttnAvindexReader,
    e_hidden: &[f32],
    layer: usize,
    kv_head: usize,
) -> Result<AvindexProbeOut, Box<dyn std::error::Error>> {
    let (cid, score) = reader.probe_embedding(e_hidden, layer, kv_head)?;
    Ok(AvindexProbeOut {
        layer,
        kv_head,
        centroid_id: cid,
        cosine: score,
    })
}

pub struct ScanStep {
    pub layer: usize,
    pub kv_head: usize,
    pub centroid_id: usize,
    pub cosine: f32,
}

pub fn scan_sidecar(
    reader: &AttnAvindexReader,
    e_hidden: &[f32],
    layers_only: Option<&[usize]>,
) -> Result<Vec<ScanStep>, Box<dyn std::error::Error>> {
    let nl = reader.num_layers();
    let mut steps = Vec::new();
    let layer_iter: Vec<usize> = if let Some(lo) = layers_only {
        lo.iter().copied().filter(|&l| l < nl).collect()
    } else {
        (0..nl).collect()
    };
    for layer in layer_iter {
        let nkv = reader.n_kv(layer);
        for kv_h in 0..nkv {
            let (cid, score) = reader.probe_embedding(e_hidden, layer, kv_h)?;
            steps.push(ScanStep {
                layer,
                kv_head: kv_h,
                centroid_id: cid,
                cosine: score,
            });
        }
    }
    Ok(steps)
}
