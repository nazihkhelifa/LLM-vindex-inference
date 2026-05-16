//! Grounded causal head pruning — dual reference vs pruned forward (Python `vindex_causal_grounded.py`).

use std::time::Instant;

use rayon::prelude::*;

use crate::kernels::{apply_rope_hf, dot, matvec_par, rms_norm_1, rms_norm_qk};
use crate::vindex::{AttnWeights, Vindex};

/// Tunables matching the Python grounded script defaults.
#[derive(Clone, Copy, Debug)]
pub struct GroundedConfig {
    pub min_pruning_k: usize,
    /// Cumulative head-probability mass required (e.g. 0.88).
    pub energy_threshold: f32,
    pub softmax_temp: f32,
}

impl Default for GroundedConfig {
    fn default() -> Self {
        Self {
            min_pruning_k: 2,
            energy_threshold: 0.88,
            softmax_temp: 4.5,
        }
    }
}

pub struct GroundedPassResult {
    pub top_k: Vec<(u32, f32)>,
    pub total_ms: f64,
    pub layer_ms: f64,
    pub logits_ms: f64,
    /// Final hidden after RMSNorm (for logit telemetry).
    pub final_state: Vec<f32>,
}

pub struct GroundedCompareResult {
    pub reference: GroundedPassResult,
    pub grounded: GroundedPassResult,
    /// Per layer: heads sorted by BOS-masked max Q·K energy (reference).
    pub oracle_heads: Vec<Vec<usize>>,
    /// Per layer: active head indices at last token (grounded).
    pub executed_heads: Vec<Vec<usize>>,
    pub head_recall_pct: f32,
    pub logit_mse: f32,
    pub logit_cosine: f32,
}

struct QkvLayer {
    all_q: Vec<Vec<f32>>,
    all_k: Vec<Vec<f32>>,
    all_v: Vec<Vec<f32>>,
}

fn rope_params(layer: usize) -> (f64, f64) {
    if (layer + 1) % 6 == 0 {
        (1e6, 8.0)
    } else {
        (1e4, 1.0)
    }
}

fn build_qkv(
    attn: &AttnWeights,
    normed: &[Vec<f32>],
    nq: usize,
    nkv: usize,
    hd: usize,
    h: usize,
    layer: usize,
) -> QkvLayer {
    let (rb, rf) = rope_params(layer);
    let mut all_q = Vec::with_capacity(normed.len());
    let mut all_k = Vec::with_capacity(normed.len());
    let mut all_v = Vec::with_capacity(normed.len());
    for (tok, x) in normed.iter().enumerate() {
        let mut q = matvec_par(&attn.w_q, x, nq * hd, h);
        let mut k = matvec_par(&attn.w_k, x, nkv * hd, h);
        let v = matvec_par(&attn.w_v, x, nkv * hd, h);
        for hi in 0..nq {
            rms_norm_qk(&mut q[hi * hd..(hi + 1) * hd], &attn.q_norm);
        }
        for hi in 0..nkv {
            rms_norm_qk(&mut k[hi * hd..(hi + 1) * hd], &attn.k_norm);
        }
        let pos = tok as f64 / rf;
        for hi in 0..nq {
            apply_rope_hf(&mut q[hi * hd..(hi + 1) * hd], pos, rb, hd);
        }
        for hi in 0..nkv {
            apply_rope_hf(&mut k[hi * hd..(hi + 1) * hd], pos, rb, hd);
        }
        all_q.push(q);
        all_k.push(k);
        all_v.push(v);
    }
    QkvLayer { all_q, all_k, all_v }
}

/// Per-head routing score: max causal Q·K over positions j>=1 (skip BOS sink at 0).
fn head_energies_sink_masked(
    qkv: &QkvLayer,
    tok: usize,
    nq: usize,
    _nkv: usize,
    hd: usize,
    groups: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut energies = vec![0.0f32; nq];
    let qrow = &qkv.all_q[tok];
    for hi in 0..nq {
        let kv_hi = hi / groups;
        let qs = hi * hd;
        let ks = kv_hi * hd;
        let mut max_e = f32::NEG_INFINITY;
        let j_start = if tok > 0 { 1 } else { 0 };
        for j in j_start..=tok {
            let e = dot(&qrow[qs..qs + hd], &qkv.all_k[j][ks..ks + hd]) * attn_scale;
            if e > max_e {
                max_e = e;
            }
        }
        energies[hi] = max_e;
    }
    energies
}

fn select_active_heads(energies: &[f32], cfg: GroundedConfig) -> (Vec<usize>, usize, f32, Vec<f32>) {
    let nq = energies.len();
    let mut order: Vec<usize> = (0..nq).collect();
    order.sort_by(|&a, &b| {
        energies[b]
            .partial_cmp(&energies[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_e = energies[order[0]];
    let exp: Vec<f32> = order
        .iter()
        .map(|&h| ((energies[h] - max_e) / cfg.softmax_temp).exp())
        .collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();
    let mut cum = 0.0f32;
    let mut chosen_k = cfg.min_pruning_k;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        chosen_k = i + 1;
        if cum >= cfg.energy_threshold {
            break;
        }
    }
    chosen_k = chosen_k.max(cfg.min_pruning_k).min(nq);
    let cov = probs.iter().take(chosen_k).sum::<f32>();
    let active: Vec<usize> = order.into_iter().take(chosen_k).collect();
    (active, chosen_k, cov, probs)
}

fn causal_attention_heads(
    qkv: &QkvLayer,
    tok: usize,
    active: &[usize],
    attn: &AttnWeights,
    h: usize,
    nq: usize,
    _nkv: usize,
    hd: usize,
    groups: usize,
    attn_scale: f32,
) -> Vec<f32> {
    let mut ho = vec![0.0f32; nq * hd];
    let active_set: std::collections::HashSet<usize> = active.iter().copied().collect();
    let qrow = &qkv.all_q[tok];
    for hi in 0..nq {
        if !active_set.contains(&hi) {
            continue;
        }
        let kv_hi = hi / groups;
        let qs = hi * hd;
        let ks = kv_hi * hd;
        let scores: Vec<f32> = (0..=tok)
            .map(|j| dot(&qrow[qs..qs + hd], &qkv.all_k[j][ks..ks + hd]) * attn_scale)
            .collect();
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exp_s.iter().sum::<f32>().max(1e-10);
        for j in 0..=tok {
            let w = exp_s[j] / sum_e;
            for d in 0..hd {
                ho[qs + d] += w * qkv.all_v[j][ks + d];
            }
        }
    }
    matvec_par(&attn.w_o, &ho, h, nq * hd)
}

fn run_ffn_layer(vindex: &Vindex, layer: usize, residuals: &mut [Vec<f32>]) {
    let pfln = vindex.norm_weights(layer, 2);
    let pfnl = vindex.norm_weights(layer, 3);
    for r in residuals.iter_mut() {
        let x = rms_norm_1(r, &pfln);
        let gs = vindex.gate_matvec(layer, &x);
        let us = vindex.up_matvec(layer, &x);
        let act: Vec<f32> = gs
            .iter()
            .zip(us.iter())
            .map(|(&g, &u)| crate::kernels::gelu_tanh(g) * u)
            .collect();
        let delta = vindex.down_matvec(layer, &act);
        let nf = rms_norm_1(&delta, &pfnl);
        for (ri, &d) in r.iter_mut().zip(nf.iter()) {
            *ri += d;
        }
    }
}

fn logits_top_k(vindex: &Vindex, state: &[f32], top_k: usize) -> (Vec<(u32, f32)>, f64) {
    let t0 = Instant::now();
    let vocab = vindex.config.vocab_size;
    let logits: Vec<f32> = (0..vocab)
        .into_par_iter()
        .map(|tid| {
            let e = vindex.embedding(tid);
            dot(state, &e)
        })
        .collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &s)| (i as u32, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);
    (indexed, ms)
}

fn full_logits(vindex: &Vindex, state: &[f32]) -> Vec<f32> {
    let vocab = vindex.config.vocab_size;
    (0..vocab)
        .into_par_iter()
        .map(|tid| {
            let e = vindex.embedding(tid);
            dot(state, &e)
        })
        .collect()
}

fn logit_mse_cosine(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len()) as f32;
    let mse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / n;
    let dot_ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 {
        dot_ab / (na * nb)
    } else {
        0.0
    };
    (mse, cos)
}

/// Reference forward: all heads; records oracle head ranking per layer (last token).
fn run_reference(
    vindex: &Vindex,
    token_ids: &[u32],
    top_k: usize,
    log_oracle_maps: bool,
) -> Result<(GroundedPassResult, Vec<Vec<usize>>), Box<dyn std::error::Error>> {
    let total = Instant::now();
    let h = vindex.config.hidden_size;
    let nl = vindex.config.num_layers;
    let (nq, nkv, hd) = vindex.attn_dims();
    let groups = nq / nkv;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let scale = vindex.config.embed_scale;

    let mut residuals: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&tid| {
            vindex
                .embedding(tid as usize)
                .iter()
                .map(|v| v * scale)
                .collect()
        })
        .collect();
    let nt = residuals.len();
    let mut oracle_heads = vec![Vec::new(); nl];

    let layer_t0 = Instant::now();
    for layer in 0..nl {
        if let Some(attn) = vindex.attn_layer(layer) {
            let iln = vindex.norm_weights(layer, 0);
            let paln = vindex.norm_weights(layer, 1);
            let normed: Vec<Vec<f32>> = residuals.iter().map(|r| rms_norm_1(r, &iln)).collect();
            let qkv = build_qkv(&attn, &normed, nq, nkv, hd, h, layer);

            let energies = head_energies_sink_masked(&qkv, nt - 1, nq, nkv, hd, groups, attn_scale);
            let mut order: Vec<usize> = (0..nq).collect();
            order.sort_by(|&a, &b| {
                energies[b]
                    .partial_cmp(&energies[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            oracle_heads[layer] = order.clone();

            if log_oracle_maps && layer > 0 {
                let max_e = energies[order[0]];
                let exp: Vec<f32> = order
                    .iter()
                    .map(|&hi| ((energies[hi] - max_e) / GroundedConfig::default().softmax_temp).exp())
                    .collect();
                let sum: f32 = exp.iter().sum();
                let parts: Vec<String> = order
                    .iter()
                    .enumerate()
                    .map(|(i, &hi)| format!("H{hi}(E:{:.2}, P:{:.1}%)", energies[hi], exp[i] / sum * 100.0))
                    .collect();
                log::info!(
                    "[L{layer}] Reference Oracle Energy Map. Window: [{}]",
                    parts.join(", ")
                );
            } else if log_oracle_maps && layer == 0 {
                log::info!("[L0] Reference Grounding Phase: All {nq}/{nq} heads systematically active.");
            }

            for tok in 0..nt {
                let active: Vec<usize> = (0..nq).collect();
                let attn_out =
                    causal_attention_heads(&qkv, tok, &active, &attn, h, nq, nkv, hd, groups, attn_scale);
                let na = rms_norm_1(&attn_out, &paln);
                for (r, a) in residuals[tok].iter_mut().zip(na.iter()) {
                    *r += a;
                }
            }
        }
        run_ffn_layer(vindex, layer, &mut residuals);
    }
    let layer_ms = layer_t0.elapsed().as_secs_f64() * 1000.0;

    let final_n = vindex.final_norm();
    let state = rms_norm_1(&residuals[nt - 1], &final_n);
    let (top_k_out, logits_ms) = logits_top_k(vindex, &state, top_k);

    Ok((
        GroundedPassResult {
            top_k: top_k_out,
            total_ms: total.elapsed().as_secs_f64() * 1000.0,
            layer_ms,
            logits_ms,
            final_state: state,
        },
        oracle_heads,
    ))
}

/// Grounded forward: layer 0 all heads; later layers prune to dynamic K.
fn run_grounded(
    vindex: &Vindex,
    token_ids: &[u32],
    top_k: usize,
    cfg: GroundedConfig,
    log_windows: bool,
) -> Result<(GroundedPassResult, Vec<Vec<usize>>), Box<dyn std::error::Error>> {
    let total = Instant::now();
    let h = vindex.config.hidden_size;
    let nl = vindex.config.num_layers;
    let (nq, nkv, hd) = vindex.attn_dims();
    let groups = nq / nkv;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let scale = vindex.config.embed_scale;

    let mut residuals: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&tid| {
            vindex
                .embedding(tid as usize)
                .iter()
                .map(|v| v * scale)
                .collect()
        })
        .collect();
    let nt = residuals.len();
    let mut executed = vec![Vec::new(); nl];

    let layer_t0 = Instant::now();
    for layer in 0..nl {
        if let Some(attn) = vindex.attn_layer(layer) {
            let iln = vindex.norm_weights(layer, 0);
            let paln = vindex.norm_weights(layer, 1);
            let normed: Vec<Vec<f32>> = residuals.iter().map(|r| rms_norm_1(r, &iln)).collect();
            let qkv = build_qkv(&attn, &normed, nq, nkv, hd, h, layer);

            for tok in 0..nt {
                let (active, chosen_k, cov, probs) = if layer == 0 {
                    let a: Vec<usize> = (0..nq).collect();
                    (a, nq, 1.0, vec![1.0 / nq as f32; nq])
                } else {
                    let energies = head_energies_sink_masked(&qkv, tok, nq, nkv, hd, groups, attn_scale);
                    select_active_heads(&energies, cfg)
                };

                if tok == nt - 1 {
                    executed[layer] = active.clone();
                    if log_windows {
                        if layer == 0 {
                            log::info!("[L0] Grounding Phase: Bypassed pruning. Executed all {nq}/{nq} heads.");
                        } else {
                            let parts: Vec<String> = active
                                .iter()
                                .enumerate()
                                .map(|(i, &hi)| {
                                    let p = probs.get(i).copied().unwrap_or(0.0);
                                    format!("H{hi}(P:{p:.1}%)")
                                })
                                .collect();
                            log::info!(
                                "[L{layer}] Grounded Causal Loop (Preserved Norm). K={chosen_k}/{nq} (Cov:{cov:.1}%). Window: [{}]",
                                parts.join(", ")
                            );
                        }
                    }
                }

                let attn_out =
                    causal_attention_heads(&qkv, tok, &active, &attn, h, nq, nkv, hd, groups, attn_scale);
                let na = rms_norm_1(&attn_out, &paln);
                for (r, a) in residuals[tok].iter_mut().zip(na.iter()) {
                    *r += a;
                }
            }
        }
        run_ffn_layer(vindex, layer, &mut residuals);
    }
    let layer_ms = layer_t0.elapsed().as_secs_f64() * 1000.0;

    let final_n = vindex.final_norm();
    let state = rms_norm_1(&residuals[nt - 1], &final_n);
    let (top_k_out, logits_ms) = logits_top_k(vindex, &state, top_k);

    Ok((
        GroundedPassResult {
            top_k: top_k_out,
            total_ms: total.elapsed().as_secs_f64() * 1000.0,
            layer_ms,
            logits_ms,
            final_state: state,
        },
        executed,
    ))
}

/// Single grounded forward (greedy conversation / top-k). No reference pass or audit logs.
pub fn infer_grounded(
    vindex: &Vindex,
    token_ids: &[u32],
    top_k: usize,
    cfg: GroundedConfig,
) -> Result<GroundedPassResult, Box<dyn std::error::Error>> {
    if vindex.attn_raw().is_none() {
        return Err(
            "grounded-causal requires attn_weights.bin (or --attn-weights).".into(),
        );
    }
    let (result, _) = run_grounded(vindex, token_ids, top_k, cfg, false)?;
    Ok(result)
}

/// Run reference + grounded paths, head-recall audit, logit MSE/cosine.
pub fn compare(
    vindex: &Vindex,
    token_ids: &[u32],
    top_k: usize,
    cfg: GroundedConfig,
) -> Result<GroundedCompareResult, Box<dyn std::error::Error>> {
    if vindex.attn_raw().is_none() {
        return Err(
            "grounded-causal requires attn_weights.bin (or --attn-weights).".into(),
        );
    }

    log::info!("STAGE 1: reference (all heads)");
    let (reference, oracle_heads) = run_reference(vindex, token_ids, top_k, true)?;

    log::info!("STAGE 2: grounded causal pruning");
    let (grounded, executed_heads) = run_grounded(vindex, token_ids, top_k, cfg, true)?;

    let nl = vindex.config.num_layers;
    let mut captured = 0usize;
    let mut slots = 0usize;
    for layer in 0..nl {
        if oracle_heads[layer].is_empty() || executed_heads[layer].is_empty() {
            continue;
        }
        let exec_set: std::collections::HashSet<usize> =
            executed_heads[layer].iter().copied().collect();
        for &h in oracle_heads[layer].iter().take(3) {
            slots += 1;
            if exec_set.contains(&h) {
                captured += 1;
            }
        }
    }
    let head_recall_pct = if slots > 0 {
        (captured as f32 / slots as f32) * 100.0
    } else {
        0.0
    };

    let ref_logits = full_logits(vindex, &reference.final_state);
    let grd_logits = full_logits(vindex, &grounded.final_state);
    let (logit_mse, logit_cosine) = logit_mse_cosine(&ref_logits, &grd_logits);

    Ok(GroundedCompareResult {
        reference,
        grounded,
        oracle_heads,
        executed_heads,
        head_recall_pct,
        logit_mse,
        logit_cosine,
    })
}

pub fn print_alignment_report(result: &GroundedCompareResult) {
    let nl = result.oracle_heads.len();
    println!();
    println!("{}", "=".repeat(85));
    println!("               GROUNDED CAUSAL ATTENTION WINDOW ALIGNMENT AUDIT");
    println!("{}", "=".repeat(85));
    println!(
        " Layer  | True Dominant Sequence (Top-3) | Final Token Causal Window    | Coverage"
    );
    println!("{}", "-".repeat(85));

    let mut captured = 0usize;
    let mut slots = 0usize;
    for layer in 0..nl {
        let oracle = &result.oracle_heads[layer];
        let exec = &result.executed_heads[layer];
        if oracle.is_empty() || exec.is_empty() {
            continue;
        }
        let hits: usize = oracle.iter().take(3).filter(|h| exec.contains(h)).count();
        captured += hits;
        slots += 3;
        let true_str: String = oracle.iter().take(3).map(|h| format!("H{h}")).collect::<Vec<_>>().join(", ");
        let exec_str: String = exec.iter().map(|h| format!("H{h}")).collect::<Vec<_>>().join(", ");
        println!(
            " L{layer:<4} | {true_str:<30} | {exec_str:<28} | {hits}/3"
        );
    }
    println!("{}", "-".repeat(85));
    println!(
        " TOTAL CRITICAL ROUTING HEADS CAPTURED: {captured} / {slots} slots"
    );
    println!(
        " MARKOV ENGINE TARGET RETENTION ACCURACY: {:.2}%",
        result.head_recall_pct
    );
    println!("{}", "=".repeat(85));
}

pub fn print_comparison_table(
    result: &GroundedCompareResult,
    decode: impl Fn(u32) -> String,
) {
    println!();
    println!("{}", "=".repeat(95));
    println!("                 TOP-10 NEXT TOKEN INFERENCE COMPARISON");
    println!("{}", "=".repeat(95));
    println!(
        " Rank |  [ENGINE A] REFERENCE (UNPRUNED)     |  [ENGINE B] GROUNDED CAUSAL ENGINE"
    );
    println!(
        "      | Token ID  | Logit Score | Decoded    | Token ID  | Logit Score | Decoded"
    );
    println!("{}", "-".repeat(95));
    let n = result
        .reference
        .top_k
        .len()
        .max(result.grounded.top_k.len());
    for idx in 0..n {
        let (r_id, r_score) = result.reference.top_k.get(idx).copied().unwrap_or((0, 0.0));
        let (m_id, m_score) = result.grounded.top_k.get(idx).copied().unwrap_or((0, 0.0));
        println!(
            "  #{:<2} |  {:<8} | {:<11.4} | {:<10} |  {:<8} | {:<11.4} | {}",
            idx + 1,
            r_id,
            r_score,
            decode(r_id),
            m_id,
            m_score,
            decode(m_id)
        );
    }
    println!("{}", "=".repeat(95));
    println!();
    println!("{}", "=".repeat(95));
    println!("   COMPARATIVE TELEMETRY");
    println!("{}", "=".repeat(95));
    println!("  * Final Logit Distribution MSE : {:.6e}", result.logit_mse);
    println!(
        "  * Cosine Vector Overlap Metric : {:.8}",
        result.logit_cosine
    );
    println!("{}", "=".repeat(95));
}
