"""
Supersaturation Taxonomy v3: Sharper measurements, shorter branches
====================================================================

Changes from v2:
1. Shorter branches (32 tokens → more branches per trace)
2. Entropy-based supersaturation (principled, less noisy than top-1 conf)
3. Pre-generation baseline supersaturation measurement per seed
4. 500 seeds (100 per discourse type) for statistical power
5. Multiple supersaturation probes per position (3, averaged)
6. Interior supersaturation: measure at sentence boundaries WITHIN branches

The v2 problem: 71% single-branch because 64-token branches exhaust the
model's "pressure" in one shot. Multi-branch traces had 2-3× higher initial
supersaturation than single-branch ones. Fix: shorter branches + better metric.

Entropy supersaturation: H(p) = -Σ p(x) log p(x) over probe tokens.
Low entropy = model is confident about continuation = HIGH supersaturation.
We invert: sat = 1 - H_norm, where H_norm = H/log(V) ∈ [0,1].
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time, math
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MASK_ID = 126336
model_name = 'GSAI-ML/LLaDA-8B-Base'

print(f"Loading {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/hf_cache',
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir='/data/hf_cache', quantization_config=bnb_config,
    device_map='auto', trust_remote_code=True,
).eval()
device = next(p.device for p in model.parameters() if p.device.type == 'cuda')

END_IDS = set()
for tok_str in ['<|eot_id|>', '<|end_of_text|>']:
    tid = tokenizer.convert_tokens_to_ids(tok_str)
    if tid is not None and tid != tokenizer.unk_token_id:
        END_IDS.add(tid)
if tokenizer.eos_token_id:
    END_IDS.add(tokenizer.eos_token_id)

VOCAB_SIZE = tokenizer.vocab_size or 128256
LOG_VOCAB = math.log(VOCAB_SIZE)

print(f"Model loaded. Vocab size: {VOCAB_SIZE}")


def extract_sentences(tokens):
    """Clean generated tokens: remove EOS, truncate to last sentence boundary."""
    clean = []
    for t in tokens:
        if t in END_IDS:
            break
        clean.append(t)
    text = tokenizer.decode(clean).strip()
    sent_ends = [m.end() for m in re.finditer(r'[.!?]+(?:\s|$)', text)]
    if sent_ends:
        text = text[:sent_ends[-1]].strip()
    elif len(text) > 30:
        text = text.strip().rstrip(',;:') + '.'
    if not text or len(text) < 5:
        return [], ''
    return tokenizer.encode(text, add_special_tokens=False), text


def internal_repetition(text, n=3):
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def denoise_branch(input_ids, start, end, steps=28, temperature=1.15):
    """Denoise masked positions iteratively."""
    for step in range(steps):
        t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float()

        mask_positions = [i for i in range(start, end) if input_ids[i] == MASK_ID]
        if not mask_positions:
            break

        cands = []
        for i in mask_positions:
            scaled = logits[i] / temperature
            probs = F.softmax(scaled, dim=-1)
            topk_probs, topk_ids = probs.topk(10)
            topk_probs = topk_probs / topk_probs.sum()
            chosen_idx = torch.multinomial(topk_probs, 1).item()
            idx = topk_ids[chosen_idx].item()
            p = probs[idx].item()
            cands.append((i, idx, p))

        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok

    return input_ids


def measure_supersaturation_entropy(text_ids, position, probe_tokens=24, n_probes=3):
    """
    Entropy-based supersaturation.
    
    Insert probe_tokens MASKs at position, measure model's entropy over each.
    Low entropy = model is confident = high supersaturation.
    
    Returns normalized supersaturation in [0, 1]:
      sat = 1 - mean(H_i / log(V))
    
    Run n_probes times and average for stability.
    """
    if len(text_ids) + probe_tokens > 2048:
        return None
    
    probe_sats = []
    
    for _ in range(n_probes):
        probed = list(text_ids[:position]) + [MASK_ID] * probe_tokens + list(text_ids[position:])
        t = torch.tensor([probed], device=device)
        
        with torch.no_grad():
            logits = model(t).logits[0].float()
        
        entropies = []
        top1_confs = []
        
        for i in range(position, position + probe_tokens):
            probs = F.softmax(logits[i], dim=-1)
            top_prob, top_id = probs.max(dim=-1)
            
            if top_id.item() in END_IDS:
                continue
            
            # Entropy: -Σ p log p
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            h_norm = entropy / LOG_VOCAB  # normalize to [0, 1]
            entropies.append(h_norm)
            top1_confs.append(top_prob.item())
        
        if entropies:
            # sat = 1 - mean(H_norm): high when entropy is low
            sat = 1.0 - (sum(entropies) / len(entropies))
            probe_sats.append(sat)
    
    if not probe_sats:
        return None
    
    avg_sat = sum(probe_sats) / len(probe_sats)
    
    return {
        'sat_entropy': round(avg_sat, 5),
        'sat_top1': round(sum(top1_confs) / max(1, len(top1_confs)), 5),
        'mean_entropy_norm': round(1.0 - avg_sat, 5),
        'n_valid': len(entropies),
    }


# ── Configuration ──
BRANCH_TOKENS = 32         # shorter branches (was 64)
MAX_BRANCHES = 20          # more branches allowed (was 12)
MAX_TOTAL_TOKENS = 640     # slightly more room
SAT_THRESHOLD = 0.5163     # calibrated from v3 run (corr 0.86 with top-1 conf)
N_PROBES = 3               # average 3 supersaturation measurements


def grow_with_trace(seed_text):
    """Grow text with shorter branches and entropy-based supersaturation."""
    prompt_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    text_ids = list(prompt_ids)
    plen = len(prompt_ids)
    trace = []
    
    # Measure INITIAL supersaturation before generating anything
    init_sat = measure_supersaturation_entropy(text_ids, len(text_ids), 24, N_PROBES)
    
    for bi in range(MAX_BRANCHES):
        if len(text_ids) - plen >= MAX_TOTAL_TOKENS:
            break
        
        bstart = len(text_ids)
        n = min(BRANCH_TOKENS, MAX_TOTAL_TOKENS - (len(text_ids) - plen))
        if n < 8:
            break
        
        ids = text_ids + [MASK_ID] * n
        ids = denoise_branch(ids, bstart, len(ids), 28, 1.15)
        
        raw = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw)
        
        if not clean_text or len(clean_tokens) < 3:
            # Measure final supersaturation even for empty branches
            sat = measure_supersaturation_entropy(text_ids, len(text_ids), 24, N_PROBES)
            trace.append({
                'branch': bi,
                'sat': sat['sat_entropy'] if sat else 0.0,
                'sat_top1': sat['sat_top1'] if sat else 0.0,
                'tokens_added': 0,
                'text': '',
                'status': 'empty',
            })
            break
        
        if internal_repetition(clean_text) > 0.35:
            sat = measure_supersaturation_entropy(text_ids, len(text_ids), 24, N_PROBES)
            trace.append({
                'branch': bi,
                'sat': sat['sat_entropy'] if sat else 0.0,
                'sat_top1': sat['sat_top1'] if sat else 0.0,
                'tokens_added': len(clean_tokens),
                'text': clean_text[:100],
                'status': 'repetitive',
            })
            break
        
        text_ids.extend(clean_tokens)
        
        # Measure supersaturation AFTER this branch
        sat = measure_supersaturation_entropy(text_ids, len(text_ids), 24, N_PROBES)
        if sat is None:
            trace.append({
                'branch': bi, 'sat': 0.0, 'sat_top1': 0.0,
                'tokens_added': len(clean_tokens), 'text': clean_text[:100],
                'status': 'overflow',
            })
            break
        
        trace.append({
            'branch': bi,
            'sat': round(sat['sat_entropy'], 5),
            'sat_top1': round(sat['sat_top1'], 5),
            'entropy_norm': round(sat['mean_entropy_norm'], 5),
            'tokens_added': len(clean_tokens),
            'text': clean_text[:100],
            'status': 'ok',
        })
        
        # Stop if supersaturation (entropy-based) drops below threshold
        if sat['sat_entropy'] < SAT_THRESHOLD:
            break
    
    final_text = tokenizer.decode(text_ids[plen:]).strip()
    return {
        'init_sat': init_sat,
        'trace': trace,
        'final_text': final_text,
        'total_tokens': len(text_ids) - plen,
        'n_branches': len([t for t in trace if t['status'] == 'ok']),
    }


# ============================================================
# SEEDS — reuse v2's 200, generate 300 more programmatically
# ============================================================

# We'll reuse the v2 seeds (loaded from file) and add 60 more per type
# For now, start with the v2 seeds and see timing
# Then we can add more in a second pass

# Load seeds from JSON — prefer 500-seed file, fall back to 200
for candidate in [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'seeds_v3_500.json'),
    '/data/seeds_v3_500.json',
    os.path.join(os.path.dirname(__file__), '..', 'data', 'seeds_v2.json'),
    '/data/seeds_v2.json',
]:
    if os.path.exists(candidate):
        seeds_json_path = candidate
        break
else:
    print("ERROR: no seeds file found")
    sys.exit(1)

with open(seeds_json_path) as f:
    seeds_raw = json.load(f)

SEEDS = {dtype: [(item[0], item[1]) for item in items] 
         for dtype, items in seeds_raw.items()}
print(f"Loaded {sum(len(v) for v in SEEDS.values())} seeds from {seeds_json_path}")


# ============================================================
# CALIBRATION RUN — find entropy threshold equivalent to old 0.10
# ============================================================

def calibrate_threshold(n_samples=10):
    """Run a few seeds to find entropy-based threshold equivalent to v2's 0.10."""
    print("\nCalibrating entropy threshold...")
    sats = []
    for dtype in ['expository', 'argumentative']:
        for label, seed_text in SEEDS[dtype][:5]:
            prompt_ids = tokenizer.encode(seed_text, add_special_tokens=False)
            # Generate one branch
            ids = list(prompt_ids) + [MASK_ID] * 64
            ids = denoise_branch(ids, len(prompt_ids), len(ids), 28, 1.15)
            clean_tokens, _ = extract_sentences(ids[len(prompt_ids):])
            if clean_tokens:
                full_ids = list(prompt_ids) + clean_tokens
                sat = measure_supersaturation_entropy(full_ids, len(full_ids), 24, 1)
                if sat:
                    sats.append((sat['sat_entropy'], sat['sat_top1']))
                    print(f"  {label[:20]:20s}: entropy_sat={sat['sat_entropy']:.4f}, top1={sat['sat_top1']:.4f}")
    
    if sats:
        # Map: what entropy_sat corresponds to top1 ≈ 0.10?
        entropy_vals = [s[0] for s in sats]
        top1_vals = [s[1] for s in sats]
        print(f"\n  Entropy sat range: {min(entropy_vals):.4f} - {max(entropy_vals):.4f}")
        print(f"  Top-1 conf range:  {min(top1_vals):.4f} - {max(top1_vals):.4f}")
        
        # Correlation
        corr = np.corrcoef(entropy_vals, top1_vals)[0, 1]
        print(f"  Correlation: {corr:.3f}")
        
        # Find entropy threshold where median top1 is ~0.10
        above = [e for e, t in sats if t >= 0.10]
        below = [e for e, t in sats if t < 0.10]
        if above and below:
            threshold = (min(above) + max(below)) / 2
        else:
            threshold = np.median(entropy_vals)
        print(f"  Suggested entropy threshold: {threshold:.4f}")
        return threshold
    return SAT_THRESHOLD


# ============================================================
# RUN
# ============================================================

def main():
    print(f"\nUsing pre-calibrated entropy threshold: {SAT_THRESHOLD:.4f}")
    
    total_seeds = sum(len(v) for v in SEEDS.values())
    print(f"\n{'=' * 80}")
    print(f"SUPERSATURATION TAXONOMY v3")
    print(f"{total_seeds} seeds across {len(SEEDS)} discourse types")
    print(f"Branch length: {BRANCH_TOKENS} tokens (was 64)")
    print(f"Threshold: {SAT_THRESHOLD:.4f} (entropy-based)")
    print(f"Max branches: {MAX_BRANCHES}")
    print(f"Probes per measurement: {N_PROBES}")
    print(f"{'=' * 80}")
    
    all_results = []
    t_start = time.time()
    
    for dtype, seeds in SEEDS.items():
        print(f"\n{'═' * 60}")
        print(f"DISCOURSE TYPE: {dtype.upper()} ({len(seeds)} seeds)")
        print(f"{'═' * 60}")
        
        for si, (label, seed_text) in enumerate(seeds):
            t0 = time.time()
            result = grow_with_trace(seed_text)
            elapsed = time.time() - t0
            
            trace_sats = [t['sat'] for t in result['trace'] if t['status'] == 'ok']
            
            # Classify shape
            if len(trace_sats) >= 2:
                diffs = [trace_sats[i+1] - trace_sats[i] for i in range(len(trace_sats)-1)]
                n_rising = sum(1 for d in diffs if d > 0.002)
                n_falling = sum(1 for d in diffs if d < -0.002)
                n_flat = len(diffs) - n_rising - n_falling
                
                if n_rising == 0 and n_falling > 0:
                    shape = 'monotone_decay'
                elif n_rising > 0 and n_falling > 0:
                    shape = 'oscillating'
                elif n_flat == len(diffs):
                    shape = 'flat'
                elif n_rising > n_falling:
                    shape = 'rising'
                else:
                    shape = 'mixed'
            elif len(trace_sats) == 1:
                shape = 'single_branch'
            else:
                shape = 'empty'
            
            entry = {
                'label': label,
                'dtype': dtype,
                'seed': seed_text[:80],
                'init_sat': result['init_sat'],
                'trace': result['trace'],
                'trace_sats': trace_sats,
                'shape': shape,
                'n_branches': result['n_branches'],
                'total_tokens': result['total_tokens'],
                'elapsed': round(elapsed, 1),
                'final_text': result['final_text'][:200],
            }
            all_results.append(entry)
            
            sat_str = ' → '.join(f'{s:.4f}' for s in trace_sats[:8])
            if len(trace_sats) > 8:
                sat_str += f' ... ({len(trace_sats)} total)'
            done = len(all_results)
            eta_min = (elapsed * (total_seeds - done)) / 60
            init_s = result['init_sat']['sat_entropy'] if result['init_sat'] else 0
            print(f"  [{done}/{total_seeds}] {label:25s}: init={init_s:.4f} | "
                  f"{result['n_branches']:2d}br, {result['total_tokens']:3d}tok, "
                  f"{shape:15s} | {sat_str} ({elapsed:.0f}s, ETA {eta_min:.0f}m)")
            
            # Save incremental
            if done % 10 == 0:
                with open('/data/sat_taxonomy_v3_partial.json', 'w') as f:
                    json.dump({
                        'results': all_results, 'done': done, 'total': total_seeds,
                        'threshold': SAT_THRESHOLD, 'branch_tokens': BRANCH_TOKENS,
                    }, f, default=str)
    
    # ── ANALYSIS ──
    elapsed_total = (time.time() - t_start) / 60
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS (completed in {elapsed_total:.0f} minutes)")
    print(f"{'=' * 80}")
    
    # 1. Branch count by type
    print("\n1. BRANCH COUNT BY DISCOURSE TYPE")
    print(f"{'Type':15s} {'N':>4s} {'AvgBr':>6s} {'Multi%':>7s} {'AvgTok':>7s} {'InitSat':>8s}")
    print("-" * 50)
    for dtype in SEEDS:
        items = [r for r in all_results if r['dtype'] == dtype]
        branches = [r['n_branches'] for r in items]
        multi = sum(1 for b in branches if b > 1)
        tokens = [r['total_tokens'] for r in items]
        init_sats = [r['init_sat']['sat_entropy'] for r in items if r['init_sat']]
        print(f"{dtype:15s} {len(items):4d} {np.mean(branches):6.1f} "
              f"{100*multi/len(items):6.1f}% {np.mean(tokens):7.0f} "
              f"{np.mean(init_sats):8.4f}")
    
    # 2. Shape distribution
    print("\n2. DECAY SHAPE DISTRIBUTION")
    for dtype in SEEDS:
        shapes = Counter([r['shape'] for r in all_results if r['dtype'] == dtype])
        total = sum(shapes.values())
        shape_str = ', '.join(f'{s}:{c}' for s, c in shapes.most_common())
        print(f"  {dtype:15s}: {shape_str}")
    
    # 3. Initial vs final supersaturation
    print("\n3. INITIAL vs FINAL SUPERSATURATION")
    for dtype in SEEDS:
        items = [r for r in all_results if r['dtype'] == dtype and r['trace_sats']]
        init = [r['init_sat']['sat_entropy'] for r in items if r['init_sat']]
        final = [r['trace_sats'][-1] for r in items if r['trace_sats']]
        if init and final:
            print(f"  {dtype:15s}: init={np.mean(init):.4f}±{np.std(init):.4f}  "
                  f"final={np.mean(final):.4f}±{np.std(final):.4f}  "
                  f"drop={np.mean(init)-np.mean(final):+.4f}")
    
    # 4. Decay rate (multi-branch only)
    print("\n4. DECAY RATE (multi-branch traces, linear slope)")
    for dtype in SEEDS:
        slopes = []
        for r in all_results:
            if r['dtype'] == dtype and len(r['trace_sats']) >= 3:
                x = list(range(len(r['trace_sats'])))
                slope = np.polyfit(x, r['trace_sats'], 1)[0]
                slopes.append(slope)
        if slopes:
            print(f"  {dtype:15s}: {np.mean(slopes):+.5f}±{np.std(slopes):.5f} (n={len(slopes)})")
    
    # 5. Notable traces
    print("\n5. NOTABLE MULTI-BRANCH TRACES")
    for r in sorted(all_results, key=lambda x: -x['n_branches'])[:10]:
        sat_str = ' → '.join(f'{s:.3f}' for s in r['trace_sats'][:6])
        print(f"  {r['label']:25s} ({r['dtype']:12s}): {r['n_branches']:2d}br | {sat_str}")
    
    # 6. Effect size: discourse type → branching
    print("\n6. DISCOURSE TYPE → BRANCHING (pairwise comparisons)")
    from scipy import stats
    dtypes = list(SEEDS.keys())
    for i in range(len(dtypes)):
        for j in range(i+1, len(dtypes)):
            a = [r['n_branches'] for r in all_results if r['dtype'] == dtypes[i]]
            b = [r['n_branches'] for r in all_results if r['dtype'] == dtypes[j]]
            t_stat, p_val = stats.ttest_ind(a, b)
            d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2)
            if p_val < 0.05:
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            else:
                sig = 'ns'
            print(f"  {dtypes[i]:14s} vs {dtypes[j]:14s}: d={d:+.3f}, p={p_val:.4f} {sig}")
    
    # Save
    outpath = '/data/sat_taxonomy_v3_results.json'
    with open(outpath, 'w') as f:
        json.dump({
            'experiment': 'supersaturation_taxonomy_v3',
            'model': model_name,
            'threshold': SAT_THRESHOLD,
            'branch_tokens': BRANCH_TOKENS,
            'n_probes': N_PROBES,
            'n_seeds': len(all_results),
            'discourse_types': list(SEEDS.keys()),
            'elapsed_minutes': round(elapsed_total, 1),
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
