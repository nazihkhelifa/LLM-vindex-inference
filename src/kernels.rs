//! Shared transformer kernels (RMSNorm, RoPE, GELU, matvec).

use rayon::prelude::*;

pub fn rms_norm_1(x: &[f32], g: &[f32]) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64 + 1e-6).sqrt() as f32;
    x.iter().zip(g.iter()).map(|(&v, &gi)| (v / rms) * (1.0 + gi)).collect()
}

pub fn rms_norm_qk(h: &mut [f32], g: &[f32]) {
    let n = h.len();
    let rms = (h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64 + 1e-6).sqrt() as f32;
    for (v, &gi) in h.iter_mut().zip(g.iter()) {
        *v = (*v / rms) * (1.0 + gi);
    }
}

pub fn apply_rope_hf(h: &mut [f32], pos: f64, base: f64, hd: usize) {
    let half = hd / 2;
    for i in 0..half {
        let f = pos / base.powf(2.0 * i as f64 / hd as f64);
        let (c, s) = (f.cos() as f32, f.sin() as f32);
        let (x1, x2) = (h[i], h[i + half]);
        h[i] = x1 * c - x2 * s;
        h[i + half] = x2 * c + x1 * s;
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

pub fn matvec_par(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .map(|r| {
            let s = r * cols;
            w[s..s + cols].iter().zip(x.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
