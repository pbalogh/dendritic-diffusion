"""
Forest Generation Test: Is Dendritic Diffusion a Stochastic L-System?
=====================================================================
Run the same seed 10 times with temperature sampling.
If the resulting structures are similar (same branch count, similar
supersaturation profiles, different surface text), then our generation
process has a "grammar" — it's a stochastic L-system, not just noise.

Key metrics to compare across runs:
  1. Number of branches
  2. Total tokens generated
  3. Supersaturation trace (sat at each branch)
  4. Branch lengths (tokens per branch)
  5. Stopping reason
  6. Surface text similarity (do different runs say different things?)
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time, math
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

STRUCTURAL_TOKENS = set()
for word in ['the','a','an','is','are','was','were','be','been','being',
             'have','has','had','do','does','did','will','would','could',
             'should','may','might','can','shall','must',
             'of','in','to','for','with','on','at','by','from','as',
             'into','through','during','before','after','between',
             'and','or','but','nor','yet','so','because','although',
             'while','when','where','if','then','than','that','which',
             'who','whom','whose','what','how','whether',
             'it','its','this','that','these','those','he','she','they',
             'his','her','their','him','them','we','our','you','your',
             'not','no','also','very','more','most','much','many',
             ',','.',':',';','!','?','-','(',')','"',"'"]:
    for variant in [word, ' ' + word]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        STRUCTURAL_TOKENS.update(ids)

print(f"Model loaded. {len(STRUCTURAL_TOKENS)} structural, {len(END_IDS)} end ids.")


def make_prompt_ids(seed_text):
    return tokenizer.encode(seed_text, add_special_tokens=False)


def get_ngrams(text, n=4):
    words = text.lower().split()
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def ngram_overlap(new_text, existing_text, n=4):
    new_grams = get_ngrams(new_text, n)
    if not new_grams:
        return 0.0
    existing_grams = get_ngrams(existing_text, n)
    overlap = new_grams & existing_grams
    return len(overlap) / len(new_grams)


def internal_repetition(text, n=3):
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def denoise_branch(input_ids, start, end, steps=28, temperature=1.15):
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
            topk_probs, topk_ids = probs.topk(5)
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


def measure_supersaturation(crystal, probe_tokens=48):
    probe_ids = list(crystal) + [MASK_ID] * probe_tokens
    t = torch.tensor([probe_ids], device=device)
    
    with torch.no_grad():
        logits = model(t).logits[0].float()
    
    measurements = []
    for i in range(len(crystal), len(probe_ids)):
        probs = F.softmax(logits[i], dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        is_end = top_id.item() in END_IDS
        structural_mass = sum(probs[tid].item() for tid in STRUCTURAL_TOKENS 
                            if tid < len(probs))
        measurements.append({
            'total_conf': top_prob.item(),
            'is_end': is_end,
            'structural_mass': structural_mass,
        })
    
    n_end = sum(1 for m in measurements if m['is_end'])
    content_meas = [m for m in measurements if not m['is_end']]
    
    if not content_meas:
        return 0.0, 0.0, 0.0, n_end / max(1, len(measurements))
    
    total_sat = sum(m['total_conf'] for m in content_meas) / len(content_meas)
    struct_sat = sum(m['structural_mass'] for m in content_meas) / len(content_meas)
    semantic_sat = 1.0 - struct_sat
    end_frac = n_end / max(1, len(measurements))
    
    return total_sat, struct_sat, semantic_sat, end_frac


def extract_sentences(tokens):
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
    clean_toks = tokenizer.encode(text, add_special_tokens=False)
    return clean_toks, text


def dendritic_generate(seed_text, temperature=1.15):
    """Single dendritic generation run. Returns structured result."""
    prompt_ids = make_prompt_ids(seed_text)
    crystal = list(prompt_ids)
    plen = len(prompt_ids)
    branches = []
    total_tokens = 0
    consecutive_empty = 0
    all_generated_text = ""
    
    branch_tokens = 64
    steps_per_branch = 28
    max_branches = 16
    max_total_tokens = 768
    
    t0 = time.time()
    
    for bi in range(max_branches):
        if total_tokens >= max_total_tokens:
            break
        
        bstart = len(crystal)
        n = min(branch_tokens, max_total_tokens - total_tokens)
        ids = crystal + [MASK_ID] * n
        
        ids = denoise_branch(ids, bstart, len(ids), steps_per_branch, temperature)
        
        raw_branch = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw_branch)
        
        if not clean_text or len(clean_tokens) < 3:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        
        consecutive_empty = 0
        
        # Repetition checks
        int_rep = internal_repetition(clean_text, n=3)
        if int_rep > 0.30:
            branches.append({
                'idx': bi, 'text': clean_text[:60], 'n_tokens': len(clean_tokens),
                'decision': 'DISCARD', 'sat': 0, 'int_rep': round(int_rep, 3),
            })
            continue
        
        if all_generated_text:
            overlap = ngram_overlap(clean_text, all_generated_text, n=4)
            if overlap > 0.40:
                branches.append({
                    'idx': bi, 'text': clean_text[:60], 'n_tokens': len(clean_tokens),
                    'decision': 'STOP_OVERLAP', 'sat': 0, 'overlap': round(overlap, 3),
                })
                break
        
        crystal.extend(clean_tokens)
        total_tokens += len(clean_tokens)
        all_generated_text += " " + clean_text
        
        total_sat, struct_sat, semantic_sat, end_frac = measure_supersaturation(crystal)
        
        decision = 'GROW'
        if total_sat < 0.03 and end_frac >= 0.99:
            decision = 'STOP_ZERO'
        elif total_sat < 0.07 and not (total_sat >= 0.07 and semantic_sat > 0.55):
            if not (total_sat >= 0.07 and struct_sat > 0.45):
                decision = 'STOP_EQUILIBRIUM'
        
        branches.append({
            'idx': bi, 'text': clean_text, 'n_tokens': len(clean_tokens),
            'decision': decision, 'sat': round(total_sat, 4),
            'struct_sat': round(struct_sat, 4), 'semantic_sat': round(semantic_sat, 4),
            'end_frac': round(end_frac, 3), 'int_rep': round(int_rep, 3),
        })
        
        if decision.startswith('STOP'):
            break
    
    elapsed = time.time() - t0
    resp_tokens = crystal[plen:]
    response = tokenizer.decode(resp_tokens).strip()
    
    # Extract structural signature
    grow_branches = [b for b in branches if b['decision'] == 'GROW' or b['decision'].startswith('STOP')]
    sat_trace = [b['sat'] for b in grow_branches]
    length_trace = [b['n_tokens'] for b in grow_branches]
    
    return {
        'text': response,
        'chars': len(response),
        'n_branches': len(grow_branches),
        'total_tokens': total_tokens,
        'elapsed': round(elapsed, 1),
        'branches': branches,
        'sat_trace': sat_trace,
        'length_trace': length_trace,
        'stop_reason': branches[-1]['decision'] if branches else 'EMPTY',
    }


def pairwise_text_similarity(texts):
    """Jaccard similarity of 4-grams between all pairs of generated texts."""
    gram_sets = [get_ngrams(t, 4) for t in texts]
    n = len(texts)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            if gram_sets[i] and gram_sets[j]:
                intersection = len(gram_sets[i] & gram_sets[j])
                union = len(gram_sets[i] | gram_sets[j])
                sims.append(intersection / union if union > 0 else 0)
            else:
                sims.append(0)
    return sims


# ============================================================
# THE FOREST: 10 runs of the same seed
# ============================================================

SEED = "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. It can be broadly divided into two main components:"

N_RUNS = 10

print(f"\n{'=' * 75}")
print(f"FOREST GENERATION TEST: {N_RUNS} runs of same seed")
print(f"Seed: \"{SEED[:70]}...\"")
print(f"Temperature: 1.15, top-k: 5")
print(f"{'=' * 75}")

forest = []

for run_idx in range(N_RUNS):
    print(f"\n  🌳 Tree {run_idx+1}/{N_RUNS}...")
    result = dendritic_generate(SEED, temperature=1.15)
    forest.append(result)
    
    sat_str = ' → '.join(f'{s:.3f}' for s in result['sat_trace'])
    len_str = ' → '.join(str(l) for l in result['length_trace'])
    
    print(f"     {result['n_branches']}B, {result['total_tokens']}tok, "
          f"{result['chars']}ch, {result['elapsed']}s")
    print(f"     Sat: [{sat_str}]")
    print(f"     Len: [{len_str}]")
    print(f"     Stop: {result['stop_reason']}")
    print(f"     \"{result['text'][:120]}...\"")


# ============================================================
# STRUCTURAL ANALYSIS
# ============================================================

print(f"\n{'=' * 75}")
print("STRUCTURAL ANALYSIS: Are these the same 'species'?")
print(f"{'=' * 75}")

# 1. Branch count distribution
branch_counts = [r['n_branches'] for r in forest]
print(f"\n  Branch counts: {branch_counts}")
print(f"  Mean: {sum(branch_counts)/len(branch_counts):.1f}, "
      f"Std: {(sum((x-sum(branch_counts)/len(branch_counts))**2 for x in branch_counts)/len(branch_counts))**0.5:.1f}")

# 2. Total tokens
tok_counts = [r['total_tokens'] for r in forest]
print(f"\n  Total tokens: {tok_counts}")
print(f"  Mean: {sum(tok_counts)/len(tok_counts):.1f}, "
      f"Std: {(sum((x-sum(tok_counts)/len(tok_counts))**2 for x in tok_counts)/len(tok_counts))**0.5:.1f}")

# 3. Supersaturation traces — compare shapes
print(f"\n  Supersaturation traces:")
max_branches = max(len(r['sat_trace']) for r in forest)
for bi in range(max_branches):
    vals = [r['sat_trace'][bi] for r in forest if bi < len(r['sat_trace'])]
    if vals:
        mean_s = sum(vals) / len(vals)
        std_s = (sum((v - mean_s)**2 for v in vals) / len(vals)) ** 0.5
        bar = '█' * int(mean_s * 50) + '░' * (50 - int(mean_s * 50))
        print(f"    B{bi}: [{bar}] {mean_s:.3f} ± {std_s:.3f} (n={len(vals)})")

# 4. Stop reasons
stop_reasons = Counter(r['stop_reason'] for r in forest)
print(f"\n  Stop reasons: {dict(stop_reasons)}")

# 5. Surface text diversity — pairwise Jaccard similarity
texts = [r['text'] for r in forest if r['text']]
if len(texts) >= 2:
    sims = pairwise_text_similarity(texts)
    mean_sim = sum(sims) / len(sims) if sims else 0
    max_sim = max(sims) if sims else 0
    min_sim = min(sims) if sims else 0
    print(f"\n  Pairwise text similarity (4-gram Jaccard):")
    print(f"    Mean: {mean_sim:.3f}, Min: {min_sim:.3f}, Max: {max_sim:.3f}")
    print(f"    (Low = diverse surface text, High = saying the same thing)")

# 6. Coefficient of variation for structure vs content
cv_branches = ((sum((x-sum(branch_counts)/len(branch_counts))**2 for x in branch_counts)/len(branch_counts))**0.5) / max(1, sum(branch_counts)/len(branch_counts))
cv_tokens = ((sum((x-sum(tok_counts)/len(tok_counts))**2 for x in tok_counts)/len(tok_counts))**0.5) / max(1, sum(tok_counts)/len(tok_counts))

print(f"\n  Coefficient of variation:")
print(f"    Branch count: {cv_branches:.2f} (low = consistent structure)")
print(f"    Token count:  {cv_tokens:.2f} (low = consistent length)")
print(f"    Text similarity: {mean_sim:.3f} (low = diverse content)")

# The L-system test:
print(f"\n  🧪 L-SYSTEM TEST:")
if cv_branches < 0.3 and mean_sim < 0.3:
    print(f"    ✅ PASS: Consistent structure (CV={cv_branches:.2f}) + diverse content (sim={mean_sim:.3f})")
    print(f"    → Generation IS a stochastic L-system!")
elif cv_branches < 0.3 and mean_sim >= 0.3:
    print(f"    ⚠️ PARTIAL: Consistent structure but repetitive content")
    print(f"    → Deterministic L-system (need more stochasticity)")
elif cv_branches >= 0.3 and mean_sim < 0.3:
    print(f"    ⚠️ PARTIAL: Diverse content but inconsistent structure")
    print(f"    → Stochastic but no grammar (just noise)")
else:
    print(f"    ❌ FAIL: Neither consistent structure nor diverse content")
    print(f"    → Not an L-system")


# ============================================================
# SAVE
# ============================================================

outpath = '/data/forest_test_results.json'
with open(outpath, 'w') as f:
    json.dump({
        'seed': SEED,
        'n_runs': N_RUNS,
        'temperature': 1.15,
        'forest': forest,
        'analysis': {
            'branch_counts': branch_counts,
            'token_counts': tok_counts,
            'cv_branches': round(cv_branches, 3),
            'cv_tokens': round(cv_tokens, 3),
            'mean_text_similarity': round(mean_sim, 3),
            'stop_reasons': dict(stop_reasons),
        },
    }, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
