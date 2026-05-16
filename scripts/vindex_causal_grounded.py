#!/usr/bin/env python3
"""
vindex_causal_grounded.py
================================================================================
Implements Grounded Causal Markov Pruning with Calibrated Energy Scales.
Fixes the winner-take-all softmax trap to restore structural stability and factual recall.
"""

import sys
import math
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger("CausalGrounded")

WORKSPACE = Path(".")
if str(WORKSPACE.resolve()) not in sys.path:
    sys.path.insert(0, str(WORKSPACE.resolve()))

try:
    import vindex_infer_python as vip
except ImportError:
    log.error("Please ensure vindex_infer_python.py is available in the local directory.")
    raise SystemExit(1)

VINDEX_DIR = Path("./gemma3-4b.vindex")
TEST_TOKENS = [2, 818, 5279, 529, 7001, 563]  # <bos> The capital of France is
TOKEN_LABELS = ["<bos>", "The", "capital", "of", "France", "is"]

MIN_PRUNING_K = 2         
ENERGY_THRESHOLD = 0.75   
SOFTMAX_TEMP = 4.5        # Calibrated scale to prevent cross-head winner-take-all collapse
TARGET_TOP_K = 10


def extract_top_predictions(vix, target_state, top_k=10, chunk_size=4096):
    vocab = vix.config.vocab_size
    logits_full = np.empty(vocab, dtype=np.float32)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        block = vix.embed[start:end].astype(np.float32, copy=False)
        logits_full[start:end] = block @ target_state
    k = min(top_k, vocab)
    partitioned_indices = np.argpartition(-logits_full, k - 1)[:k]
    sorted_order = np.argsort(-logits_full[partitioned_indices])
    return [(int(partitioned_indices[o]), float(logits_full[partitioned_indices[o]])) for o in sorted_order]


def run_reference_inference(vix):
    cfg = vix.config
    nl = cfg.num_layers
    nq, nkv, hd = vix.attn_dims()
    groups = nq // nkv
    attn_scale = 1.0 / math.sqrt(hd)
    scale = cfg.embed_scale

    residuals = [(vix.embedding_f32(int(tid)) * scale).astype(np.float32) for tid in TEST_TOKENS]
    nt = len(residuals)
    true_heads_per_layer = {}

    for layer in range(nl):
        pfln = vix.norm_weights(layer, 2)
        pfnl = vix.norm_weights(layer, 3)
        is_global = (layer + 1) % 6 == 0
        rb, rf = (1e6, 8.0) if is_global else (1e4, 1.0)

        iln = vix.norm_weights(layer, 0)
        paln = vix.norm_weights(layer, 1)
        attn = vix.attn_layer_views(layer)

        if attn is not None:
            w_q, w_k, w_v, w_o = attn["w_q"], attn["w_k"], attn["w_v"], attn["w_o"]
            q_norm, k_norm = attn["q_norm"], attn["k_norm"]

            normed = [vip.rms_norm_1(residuals[tok], iln) for tok in range(nt)]
            all_q, all_k, all_v = [], [], []
            
            for tok in range(nt):
                x_tok = normed[tok].astype(np.float32, copy=False)
                q = (w_q @ x_tok).copy()
                k = (w_k @ x_tok).copy()
                vv = w_v @ x_tok
                
                for hi in range(nq):
                    vip.rms_norm_qk(q[hi * hd : (hi + 1) * hd], q_norm)
                for hi in range(nkv):
                    vip.rms_norm_qk(k[hi * hd : (hi + 1) * hd], k_norm)
                    
                pos = tok / rf
                for hi in range(nq):
                    vip.apply_rope_hf(q[hi * hd : (hi + 1) * hd], pos, rb, hd)
                for hi in range(nkv):
                    vip.apply_rope_hf(k[hi * hd : (hi + 1) * hd], pos, rb, hd)
                    
                all_q.append(q)
                all_k.append(k)
                all_v.append(vv)

            target_tok_idx = nt - 1
            layer_true_energies = np.zeros(nq, dtype=np.float32)
            
            for hi in range(nq):
                kv_hi = hi // groups
                qs, ks = hi * hd, kv_hi * hd
                qrow = all_q[target_tok_idx]
                
                scores = np.empty(nt, dtype=np.float32)
                for j in range(nt):
                    scores[j] = float(np.sum(qrow[qs : qs + hd] * all_k[j][ks : ks + hd])) * attn_scale
                
                if len(scores) > 1:
                    layer_true_energies[hi] = float(np.max(scores[1:]))
                else:
                    layer_true_energies[hi] = float(np.max(scores))
            
            sorted_true_heads = np.argsort(layer_true_energies)[::-1]
            true_heads_per_layer[layer] = list(sorted_true_heads)

            for tok in range(nt):
                ho = np.zeros(nq * hd, dtype=np.float32)
                for hi in range(nq):
                    kv_hi = hi // groups
                    qs, ks = hi * hd, kv_hi * hd
                    qrow = all_q[tok]
                    
                    scores = np.empty(tok + 1, dtype=np.float32)
                    for j in range(tok + 1):
                        scores[j] = float(np.sum(qrow[qs : qs + hd] * all_k[j][ks : ks + hd])) * attn_scale
                    
                    max_s = float(np.max(scores))
                    exp_s = np.exp(scores - max_s)
                    sum_e = max(float(np.sum(exp_s)), 1e-10)

                    for j in range(tok + 1):
                        ho[qs : qs + hd] += (exp_s[j] / sum_e) * all_v[j][ks : ks + hd]
                
                residuals[tok] = residuals[tok] + vip.rms_norm_1(w_o @ ho, paln)

        for tok in range(nt):
            x_tok = vip.rms_norm_1(residuals[tok], pfln)
            gs = vix.gate_matvec(layer, x_tok)
            us = vix.up_matvec(layer, x_tok)
            act = vip.gelu_tanh_vec(gs) * us
            delta = vix.down_matvec(layer, act)
            residuals[tok] = residuals[tok] + vip.rms_norm_1(delta, pfnl)

    final_ln = vix.norm_weights(nl, 0)
    target_state = vip.rms_norm_1(residuals[-1], final_ln)
    return extract_top_predictions(vix, target_state, top_k=TARGET_TOP_K), true_heads_per_layer


def run_markov_inference(vix):
    cfg = vix.config
    nl = cfg.num_layers
    nq, nkv, hd = vix.attn_dims()
    groups = nq // nkv
    attn_scale = 1.0 / math.sqrt(hd)
    scale = cfg.embed_scale

    residuals = [(vix.embedding_f32(int(tid)) * scale).astype(np.float32) for tid in TEST_TOKENS]
    nt = len(residuals)
    executed_heads_at_final_token = {}

    for layer in range(nl):
        pfln = vix.norm_weights(layer, 2)
        pfnl = vix.norm_weights(layer, 3)
        is_global = (layer + 1) % 6 == 0
        rb, rf = (1e6, 8.0) if is_global else (1e4, 1.0)

        iln = vix.norm_weights(layer, 0)
        paln = vix.norm_weights(layer, 1)
        attn = vix.attn_layer_views(layer)

        if attn is not None:
            w_q, w_k, w_v, w_o = attn["w_q"], attn["w_k"], attn["w_v"], attn["w_o"]
            q_norm, k_norm = attn["q_norm"], attn["k_norm"]

            normed = [vip.rms_norm_1(residuals[tok], iln) for tok in range(nt)]
            all_q, all_k, all_v = [], [], []
            
            for tok in range(nt):
                x_tok = normed[tok].astype(np.float32, copy=False)
                q = (w_q @ x_tok).copy()
                k = (w_k @ x_tok).copy()
                vv = w_v @ x_tok
                
                for hi in range(nq):
                    vip.rms_norm_qk(q[hi * hd : (hi + 1) * hd], q_norm)
                for hi in range(nkv):
                    vip.rms_norm_qk(k[hi * hd : (hi + 1) * hd], k_norm)
                    
                pos = tok / rf
                for hi in range(nq):
                    vip.apply_rope_hf(q[hi * hd : (hi + 1) * hd], pos, rb, hd)
                for hi in range(nkv):
                    vip.apply_rope_hf(k[hi * hd : (hi + 1) * hd], pos, rb, hd)
                    
                all_q.append(q)
                all_k.append(k)
                all_v.append(vv)

            for tok in range(nt):
                if layer == 0:
                    active_heads = list(range(nq))
                    cum_probs = np.ones(nq)
                    chosen_k = nq
                    probs = np.ones(nq) / nq
                else:
                    upcoming_energies = np.zeros(nq, dtype=np.float32)
                    qrow = all_q[tok]
                    
                    for hj in range(nq):
                        kv_hj = hj // groups
                        qs, ks = hj * hd, kv_hj * hd
                        
                        seq_scores = np.empty(tok + 1, dtype=np.float32)
                        for j in range(tok + 1):
                            seq_scores[j] = float(np.sum(qrow[qs : qs + hd] * all_k[j][ks : ks + hd])) * attn_scale
                            
                        if len(seq_scores) > 1:
                            upcoming_energies[hj] = float(np.max(seq_scores[1:]))
                        else:
                            upcoming_energies[hj] = float(np.max(seq_scores))
                    
                    sorted_next_heads = np.argsort(upcoming_energies)[::-1]
                    sorted_energies = upcoming_energies[sorted_next_heads]
                    
                    # Apply cross-head calibration scale factor
                    exp_energies = np.exp((sorted_energies - np.max(sorted_energies)) / SOFTMAX_TEMP)
                    probs = exp_energies / np.sum(exp_energies)
                    cum_probs = np.cumsum(probs)
                    
                    chosen_k = max(MIN_PRUNING_K, int(np.searchsorted(cum_probs, ENERGY_THRESHOLD)) + 1)
                    active_heads = list(sorted_next_heads[:chosen_k])

                if tok == nt - 1:
                    executed_heads_at_final_token[layer] = list(active_heads)
                    if layer == 0:
                        log.info(f"[L0] Grounding Phase: Bypassed pruning. Executed all {nq}/{nq} heads.")
                    else:
                        top_heads_str = ", ".join(f"H{h}(P:{probs[i]*100:.1f}%)" for i, h in enumerate(active_heads))
                        log.info(f"[L{layer}] Grounded Causal Loop (Preserved Norm). K={chosen_k}/{nq} (Cov:{cum_probs[chosen_k-1]*100:.1f}%). Window: [{top_heads_str}]")

                ho = np.zeros(nq * hd, dtype=np.float32)
                for hi in range(nq):
                    if hi not in active_heads:
                        continue
                        
                    kv_hi = hi // groups
                    qs, ks = hi * hd, kv_hi * hd
                    qrow = all_q[tok]
                    
                    scores = np.empty(tok + 1, dtype=np.float32)
                    for j in range(tok + 1):
                        scores[j] = float(np.sum(qrow[qs : qs + hd] * all_k[j][ks : ks + hd])) * attn_scale
                    
                    max_s = float(np.max(scores))
                    exp_s = np.exp(scores - max_s)
                    sum_e = max(float(np.sum(exp_s)), 1e-10)

                    for j in range(tok + 1):
                        ho[qs : qs + hd] += (exp_s[j] / sum_e) * all_v[j][ks : ks + hd]
                
                projected_delta = w_o @ ho
                normed_delta = vip.rms_norm_1(projected_delta, paln)
                # Structural base preservation without artificial vector scaling
                residuals[tok] = residuals[tok] + normed_delta

        for tok in range(nt):
            x_tok = vip.rms_norm_1(residuals[tok], pfln)
            gs = vix.gate_matvec(layer, x_tok)
            us = vix.up_matvec(layer, x_tok)
            act = vip.gelu_tanh_vec(gs) * us
            delta = vix.down_matvec(layer, act)
            residuals[tok] = residuals[tok] + vip.rms_norm_1(delta, pfnl)

    final_ln = vix.norm_weights(nl, 0)
    target_state = vip.rms_norm_1(residuals[-1], final_ln)
    return extract_top_predictions(vix, target_state, top_k=TARGET_TOP_K), executed_heads_at_final_token


def print_accuracy_report(true_profiles, executed_profiles, nl):
    print("\n" + "=" * 85)
    print("               GROUNDED CAUSAL ATTENTION WINDOW ALIGNMENT AUDIT")
    print("=" * 85)
    print(" Layer  | True Dominant Sequence (Top-3) | Final Token Causal Window    | Coverage")
    print("-" * 85)
    
    total_true_captured = 0
    total_possible_slots = 0
    
    for layer in range(nl):
        if layer not in true_profiles or layer not in executed_profiles:
            continue
        true_top_3 = true_profiles[layer][:3]
        executed_set = executed_profiles[layer]
        
        matches = [h for h in true_top_3 if h in executed_set]
        hit_count = len(matches)
        
        total_true_captured += hit_count
        total_possible_slots += 3
        
        true_str = ", ".join(f"H{h}" for h in true_top_3)
        exec_str = ", ".join(f"H{h}" for h in executed_set)
        
        print(f" L{layer:<4} | {true_str:<30} | {exec_str:<28} | {hit_count}/3")
        
    accuracy = (total_true_captured / total_possible_slots) * 100
    print("-" * 85)
    print(f" TOTAL CRITICAL ROUTING HEADS CAPTURED: {total_true_captured} / {total_possible_slots} slots")
    print(f" MARKOV ENGINE TARGET RETENTION ACCURACY: {accuracy:.2f}%")
    print("=" * 85)


def print_comparison_table(ref_results, markov_results, tokenizer=None):
    print("\n" + "=" * 95)
    print("                 TOP-10 NEXT TOKEN REAL-TIME INFERENCE COMPARISON")
    print("=" * 95)
    print(f" Prompt: {' '.join(TOKEN_LABELS)}")
    print("-" * 95)
    print(" Rank |  [ENGINE A] REFERENCE (UNPRUNED)     |  [ENGINE B] GROUNDED MARKOV ENGINE")
    print("      | Token ID  | Logit Score | Decoded String | Token ID  | Logit Score | Decoded String")
    print("-" * 95)
    
    for idx in range(TARGET_TOP_K):
        r_id, r_score = ref_results[idx]
        m_id, m_score = markov_results[idx]
        
        r_str, m_str = f"<{r_id}>", f"<{m_id}>"
        if tokenizer is not None:
            try:
                r_str = tokenizer.decode([r_id]).replace("\n", "\\n").replace(" ", "·")
                m_str = tokenizer.decode([m_id]).replace("\n", "\\n").replace(" ", "·")
            except Exception:
                pass
                
        print(f"  #{idx+1:<2} |  {r_id:<8} | {r_score:<11.4f} | {r_str:<14} |  {m_id:<8} | {m_score:<11.4f} | {m_str}")
    print("=" * 95)


def main():
    if not VINDEX_DIR.exists():
        log.error(f"Target folder missing: {VINDEX_DIR.resolve()}")
        return

    tokenizer_instance = None
    try:
        from tokenizers import Tokenizer
        tok_json = VINDEX_DIR / "tokenizer.json"
        if tok_json.is_file():
            tokenizer_instance = Tokenizer.from_file(str(tok_json))
            log.info("Successfully bound local token decoder module.")
    except ImportError:
        log.warning("Python 'tokenizers' dependency not found. Displaying raw tokens.")

    vix = vip.Vindex.load(VINDEX_DIR)
    
    print("\n--- STAGE 1: GATHERING ORACLE STREAM MAPS ---")
    ref_top_k, true_head_profiles = run_reference_inference(vix)
    
    print("\n--- STAGE 2: EXECUTING TRUE LOCAL INTERLEAVED LOOK-AHEAD PASS ---")
    markov_top_k, executed_profiles = run_markov_inference(vix)
    
    print_accuracy_report(true_head_profiles, executed_profiles, vix.config.num_layers)
    print_comparison_table(ref_top_k, markov_top_k, tokenizer=tokenizer_instance)


if __name__ == "__main__":
    main()