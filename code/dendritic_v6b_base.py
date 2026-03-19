"""
Dendritic Diffusion v6b: Base Model + Anti-Repetition
======================================================
v6 proved LLaDA-8B-Base eliminates EOS bias (0% across all prompts).
Two remaining problems:
  1. Over-cautious: stops after 1 branch at 200-300 chars (4/6 seeds)
  2. Repetition trap: numbered lists sustain high sat while looping (2/6)

v6b fixes:
  - Mild temperature (1.15) during denoising — just enough diversity
    to break greedy repetition, not the aggressive 1.5 from v5e
  - Cross-branch repetition detection via n-gram overlap:
    if a branch shares >40% of its 4-grams with all previous text, STOP
  - Within-branch repetition: if >30% of 3-grams are repeated, discard
  - Lower sat threshold to 0.07 — let cautious seeds continue
  - No EOS penalty needed (Base model has no EOS bias)
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time
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


# ============================================================
# REPETITION DETECTION
# ============================================================

def get_ngrams(text, n=4):
    """Extract n-grams from text (lowercased words)."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def ngram_overlap(new_text, existing_text, n=4):
    """
    What fraction of new_text's n-grams already appear in existing_text?
    Returns 0.0 (completely novel) to 1.0 (completely repeated).
    """
    new_grams = get_ngrams(new_text, n)
    if not new_grams:
        return 0.0
    existing_grams = get_ngrams(existing_text, n)
    overlap = new_grams & existing_grams
    return len(overlap) / len(new_grams)


def internal_repetition(text, n=3):
    """
    What fraction of n-grams within this text are repeated?
    High = degenerate looping text.
    """
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


# ============================================================
# DENOISING — mild temperature
# ============================================================

def denoise_branch(input_ids, start, end, steps=28, temperature=1.15):
    """
    Denoising with mild temperature to break repetition.
    temperature=1.15: just enough randomness to avoid greedy loops.
    Still picks the argmax, but logits are softened so alternatives
    have a chance when confidence is low.
    """
    step_log = []
    
    for step in range(steps):
        t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float()
        
        mask_positions = [i for i in range(start, end) if input_ids[i] == MASK_ID]
        if not mask_positions:
            break
        
        cands = []
        n_eos = 0
        for i in mask_positions:
            scaled = logits[i] / temperature
            probs = F.softmax(scaled, dim=-1)
            
            # Sample from top-k=5 instead of pure argmax
            # This gives mild diversity without going off the rails
            topk_probs, topk_ids = probs.topk(5)
            # Renormalize top-k
            topk_probs = topk_probs / topk_probs.sum()
            chosen_idx = torch.multinomial(topk_probs, 1).item()
            idx = topk_ids[chosen_idx].item()
            p = probs[idx].item()
            
            if idx in END_IDS:
                n_eos += 1
            cands.append((i, idx, p))
        
        eos_frac = n_eos / max(1, len(mask_positions))
        step_log.append({
            'step': step,
            'eos_frac': round(eos_frac, 3),
            'n_masked': len(mask_positions),
        })
        
        # Unmask by confidence
        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok
    
    return input_ids, step_log


def denoise_branch_greedy(input_ids, start, end, steps=28):
    """Pure greedy for comparison."""
    step_log = []
    for step in range(steps):
        t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float()
        
        mask_positions = [i for i in range(start, end) if input_ids[i] == MASK_ID]
        if not mask_positions:
            break
        
        cands = []
        n_eos = 0
        for i in mask_positions:
            probs = F.softmax(logits[i], dim=-1)
            p, idx = probs.max(dim=-1)
            if idx.item() in END_IDS:
                n_eos += 1
            cands.append((i, idx.item(), p.item()))
        
        eos_frac = n_eos / max(1, len(mask_positions))
        step_log.append({'step': step, 'eos_frac': round(eos_frac, 3), 'n_masked': len(mask_positions)})
        
        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok
    
    return input_ids, step_log


# ============================================================
# SUPERSATURATION
# ============================================================

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
            'top_token': tokenizer.decode([top_id.item()]).strip(),
            'top_id': top_id.item(),
            'is_end': is_end,
            'structural_mass': structural_mass,
            'semantic_mass': 1.0 - structural_mass,
        })
    
    n_end = sum(1 for m in measurements if m['is_end'])
    content_meas = [m for m in measurements if not m['is_end']]
    
    if not content_meas:
        return 0.0, 0.0, 0.0, [], n_end / max(1, len(measurements))
    
    total_sat = sum(m['total_conf'] for m in content_meas) / len(content_meas)
    struct_sat = sum(m['structural_mass'] for m in content_meas) / len(content_meas)
    semantic_sat = sum(m['semantic_mass'] for m in content_meas) / len(content_meas)
    
    top_preds = sorted(content_meas, key=lambda m: -m['total_conf'])[:5]
    end_frac = n_end / max(1, len(measurements))
    
    return total_sat, struct_sat, semantic_sat, top_preds, end_frac


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


# ============================================================
# DENDRITIC GENERATION with anti-repetition
# ============================================================

def dendritic_generate(seed_text, branch_tokens=64, steps_per_branch=28,
                       max_branches=16, max_total_tokens=768,
                       temperature=1.15, use_greedy=False,
                       overlap_threshold=0.40, internal_rep_threshold=0.30,
                       sat_threshold=0.07):
    """
    Dendritic generation with anti-repetition:
      - Cross-branch n-gram overlap detection
      - Within-branch repetition detection
      - Mild temperature sampling (default 1.15)
      - Lower sat threshold (0.07 vs 0.10 in v6)
    """
    prompt_ids = make_prompt_ids(seed_text)
    crystal = list(prompt_ids)
    plen = len(prompt_ids)
    branches = []
    total_tokens = 0
    consecutive_empty = 0
    all_step_logs = []
    
    # Track all generated text for cross-branch overlap
    all_generated_text = ""
    
    t0 = time.time()
    
    for bi in range(max_branches):
        if total_tokens >= max_total_tokens:
            break
        
        bstart = len(crystal)
        n = min(branch_tokens, max_total_tokens - total_tokens)
        ids = crystal + [MASK_ID] * n
        
        if use_greedy:
            ids, step_log = denoise_branch_greedy(ids, bstart, len(ids), steps_per_branch)
        else:
            ids, step_log = denoise_branch(ids, bstart, len(ids), steps_per_branch, temperature)
        all_step_logs.append(step_log)
        
        raw_branch = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw_branch)
        
        if not clean_text or len(clean_tokens) < 3:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        
        consecutive_empty = 0
        
        # ============ REPETITION CHECKS ============
        
        # Check 1: Internal repetition (within this branch)
        int_rep = internal_repetition(clean_text, n=3)
        if int_rep > internal_rep_threshold:
            branch = {
                'idx': bi,
                'text': clean_text[:100] + '...',
                'n_tokens': len(clean_tokens),
                'decision': f'DISCARD (internal repetition {int_rep:.0%} > {internal_rep_threshold:.0%})',
                'repetition': {'internal': round(int_rep, 3), 'overlap': None},
                'supersaturation': {'total': 0, 'structural': 0, 'semantic': 0, 'end_fraction': 0},
                'top_preds': [],
            }
            branches.append(branch)
            # Don't add to crystal, but don't count as empty either
            continue
        
        # Check 2: Cross-branch overlap (this branch vs all previous)
        if all_generated_text:
            overlap = ngram_overlap(clean_text, all_generated_text, n=4)
            if overlap > overlap_threshold:
                branch = {
                    'idx': bi,
                    'text': clean_text[:100] + '...',
                    'n_tokens': len(clean_tokens),
                    'decision': f'STOP (cross-branch overlap {overlap:.0%} > {overlap_threshold:.0%})',
                    'repetition': {'internal': round(int_rep, 3), 'overlap': round(overlap, 3)},
                    'supersaturation': {'total': 0, 'structural': 0, 'semantic': 0, 'end_fraction': 0},
                    'top_preds': [],
                }
                branches.append(branch)
                break  # Stop generation entirely
        
        # Passed repetition checks — add to crystal
        crystal.extend(clean_tokens)
        total_tokens += len(clean_tokens)
        all_generated_text += " " + clean_text
        
        # Measure supersaturation
        total_sat, struct_sat, semantic_sat, top_preds, end_frac = \
            measure_supersaturation(crystal)
        
        overlap_val = ngram_overlap(clean_text, all_generated_text, n=4) if bi > 0 else 0.0
        
        branch = {
            'idx': bi,
            'text': clean_text,
            'n_tokens': len(clean_tokens),
            'supersaturation': {
                'total': round(total_sat, 4),
                'structural': round(struct_sat, 4),
                'semantic': round(semantic_sat, 4),
                'end_fraction': round(end_frac, 3),
            },
            'top_preds': [(m['top_token'], round(m['total_conf'], 3)) 
                          for m in top_preds[:3]] if top_preds else [],
            'repetition': {
                'internal': round(int_rep, 3),
                'overlap': round(overlap_val, 3),
            },
        }
        
        # Supersaturation-based stopping (lower threshold than v6)
        if total_sat < 0.03 and end_frac >= 0.99:
            branch['decision'] = f'STOP (zero energy)'
            branches.append(branch)
            break
        
        if total_sat >= 0.25:
            branch['decision'] = f'GROW (sat={total_sat:.3f} — high energy)'
            branches.append(branch)
            continue
        
        if total_sat >= sat_threshold and semantic_sat > 0.55:
            branch['decision'] = f'GROW (sat={total_sat:.3f}, sem={semantic_sat:.2f})'
            branches.append(branch)
            continue
        
        if total_sat >= sat_threshold and struct_sat > 0.45:
            branch['decision'] = f'GROW (sat={total_sat:.3f}, struct={struct_sat:.2f})'
            branches.append(branch)
            continue
        
        branch['decision'] = f'STOP (sat={total_sat:.3f} — equilibrium)'
        branches.append(branch)
        break
    
    elapsed = time.time() - t0
    resp_tokens = crystal[plen:]
    response = tokenizer.decode(resp_tokens).strip()
    
    return response, elapsed, branches, all_step_logs


def breadth_first(seed_text, max_tokens=768, steps=64, temperature=1.15, use_greedy=False):
    prompt_ids = make_prompt_ids(seed_text)
    plen = len(prompt_ids)
    ids = prompt_ids + [MASK_ID] * max_tokens
    
    t0 = time.time()
    if use_greedy:
        ids, log = denoise_branch_greedy(ids, plen, len(ids), steps)
    else:
        ids, log = denoise_branch(ids, plen, len(ids), steps, temperature)
    elapsed = time.time() - t0
    
    resp = []
    for t in ids[plen:]:
        if t in END_IDS:
            break
        resp.append(t)
    
    return tokenizer.decode(resp).strip(), elapsed, log


def check_quality(text):
    issues = []
    words = text.split()
    if len(words) < 5:
        return ['too short']
    for i in range(len(words)-1):
        if words[i].lower() == words[i+1].lower() and words[i].lower() not in {
            'the','a','had','very','that','is','and'}:
            issues.append(f"repeated '{words[i]}'")
    trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words)-2)]
    seen = {}
    for tg in trigrams:
        seen[tg] = seen.get(tg, 0) + 1
    for tg, cnt in seen.items():
        if cnt >= 3 and tg not in {'of the the','in the the','the united states','as well as'}:
            issues.append(f"repeated '{tg}' x{cnt}")
    return issues


def count_sentences(text):
    return len([s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 5])


# Same seeds as v6
seeds = [
    ("Immune System",
     "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. It can be broadly divided into two main components:"),
    ("Apollo 11",
     "On July 16, 1969, the Saturn V rocket carrying the Apollo 11 crew lifted off from Kennedy Space Center. Commander Neil Armstrong, Lunar Module Pilot Buzz Aldrin, and Command Module Pilot Michael Collins"),
    ("French Revolution",
     "The French Revolution of 1789 was the product of decades of social inequality, economic crisis, and political dysfunction. The causes can be traced to several interrelated factors:"),
    ("Evolution",
     "Evolution by natural selection, first proposed by Charles Darwin and Alfred Russel Wallace, is the process by which populations of organisms change over successive generations. The mechanism works through several key principles:"),
    ("Government Types",
     "Throughout history, human societies have organized themselves under various forms of government. Three of the most significant are democracy, monarchy, and authoritarianism, each with distinct characteristics:"),
    ("Climate Change",
     "Climate change refers to the long-term shift in global temperatures and weather patterns. While climate has changed naturally throughout Earth's history, the current rapid warming is primarily driven by human activities, particularly"),
]


# ============================================================
# RUN 1: v6b with temperature + anti-repetition
# ============================================================

print(f"\n{'=' * 75}")
print("v6b: LLaDA-8B-Base + temp=1.15 + top-k=5 + anti-repetition")
print(f"{'=' * 75}")

v6b_results = []

for label, seed_text in seeds:
    print(f"\n{'═' * 75}")
    print(f"SEED [{label}]: \"{seed_text[:70]}...\"")
    print(f"{'═' * 75}")
    
    # BF with temperature
    bf_text, bf_time, bf_log = breadth_first(seed_text, max_tokens=768, steps=64, temperature=1.15)
    bf_issues = check_quality(bf_text)
    bf_sents = count_sentences(bf_text)
    
    # DD with temperature + anti-repetition
    dd_text, dd_time, dd_branches, dd_logs = dendritic_generate(
        seed_text, branch_tokens=64, steps_per_branch=28,
        max_branches=16, max_total_tokens=768,
        temperature=1.15,
        overlap_threshold=0.40, internal_rep_threshold=0.30,
        sat_threshold=0.07,
    )
    dd_issues = check_quality(dd_text)
    dd_sents = count_sentences(dd_text)
    
    print(f"\n  BF ({bf_sents} sent, {len(bf_text)} chars, {bf_time:.1f}s):")
    for line in bf_text[:500].split('\n'):
        print(f"    {line}")
    if len(bf_text) > 500:
        print(f"    [... +{len(bf_text)-500} chars]")
    print(f"    {'✅ Clean' if not bf_issues else f'⚠️  {len(bf_issues)} issues: {", ".join(bf_issues[:3])}'}")
    if bf_log:
        eos_fracs = [e['eos_frac'] for e in bf_log[:5]]
        print(f"    📊 EOS steps 0-4: {[f'{f:.0%}' for f in eos_fracs]}")
    
    print(f"\n  DD ({dd_sents} sent, {len(dd_branches)} branches, {len(dd_text)} chars, {dd_time:.1f}s):")
    for line in dd_text[:800].split('\n'):
        print(f"    {line}")
    if len(dd_text) > 800:
        print(f"    [... +{len(dd_text)-800} chars]")
    print(f"    {'✅ Clean' if not dd_issues else f'⚠️  {len(dd_issues)} issues: {", ".join(dd_issues[:5])}'}")
    
    print(f"\n  📊 Crystal growth:")
    for b in dd_branches:
        sat = b['supersaturation']
        rep = b.get('repetition', {})
        rep_str = f"int={rep.get('internal',0):.0%}"
        if rep.get('overlap') is not None:
            rep_str += f" ovl={rep['overlap']:.0%}"
        print(f"    B{b['idx']}: sat={sat['total']:.3f} end={sat['end_fraction']:.0%} "
              f"rep=[{rep_str}] → {b['decision'][:50]}")
        print(f"         \"{b['text'][:80]}{'...' if len(b['text'])>80 else ''}\"")
    
    if dd_logs and dd_logs[0]:
        eos_trace = [e['eos_frac'] for e in dd_logs[0][:6]]
        print(f"  🔬 B0 EOS: {[f'{f:.0%}' for f in eos_trace]}")
    
    v6b_results.append({
        'label': label, 'seed': seed_text,
        'bf': {'text': bf_text, 'time': bf_time, 'sents': bf_sents,
               'issues': bf_issues, 'chars': len(bf_text)},
        'dd': {'text': dd_text, 'time': dd_time, 'sents': dd_sents,
               'issues': dd_issues, 'chars': len(dd_text),
               'branches': dd_branches},
    })


# ============================================================
# RUN 2: v6 greedy (same as before, for comparison)
# ============================================================

print(f"\n{'=' * 75}")
print("v6 GREEDY BASELINE (same as v6, for comparison)")
print(f"{'=' * 75}")

v6_results = []

for label, seed_text in seeds:
    dd_text, dd_time, dd_branches, _ = dendritic_generate(
        seed_text, branch_tokens=64, steps_per_branch=28,
        max_branches=16, max_total_tokens=768,
        use_greedy=True,
        overlap_threshold=0.40, internal_rep_threshold=0.30,
        sat_threshold=0.07,
    )
    dd_issues = check_quality(dd_text)
    dd_sents = count_sentences(dd_text)
    
    # Show just summary
    rep_stops = sum(1 for b in dd_branches if 'overlap' in b.get('decision','') or 'repetition' in b.get('decision',''))
    print(f"  {label:<20s}: {dd_sents}S {len(dd_branches)}B {len(dd_text):>5d}ch "
          f"{'✅' if not dd_issues else f'⚠️{len(dd_issues)}'} "
          f"{'(🛑 rep detected)' if rep_stops else ''}")
    if dd_text:
        print(f"    \"{dd_text[:120]}...\"")
    
    v6_results.append({
        'label': label,
        'dd': {'text': dd_text, 'sents': dd_sents, 'issues': dd_issues,
               'chars': len(dd_text), 'branches': dd_branches},
    })


# ============================================================
# COMPARISON SUMMARY
# ============================================================

print(f"\n{'=' * 75}")
print("COMPARISON: v6b (temp+anti-rep) vs v6 (greedy+anti-rep)")
print(f"{'=' * 75}")

n = len(seeds)

print(f"\n  {'Seed':<20s} {'v6 greedy':>12s} {'v6b temp':>12s} {'v6 Q':>8s} {'v6b Q':>8s} {'v6 B':>5s} {'v6b B':>5s}")
print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*5} {'─'*5}")
for i in range(n):
    label = seeds[i][0]
    g_ch = v6_results[i]['dd']['chars']
    t_ch = v6b_results[i]['dd']['chars']
    g_q = '✅' if not v6_results[i]['dd']['issues'] else f"⚠️{len(v6_results[i]['dd']['issues'])}"
    t_q = '✅' if not v6b_results[i]['dd']['issues'] else f"⚠️{len(v6b_results[i]['dd']['issues'])}"
    g_b = len(v6_results[i]['dd']['branches'])
    t_b = len(v6b_results[i]['dd']['branches'])
    print(f"  {label:<20s} {g_ch:>10d}ch {t_ch:>10d}ch {g_q:>8s} {t_q:>8s} {g_b:>5d} {t_b:>5d}")

# Averages
g_avg = sum(r['dd']['chars'] for r in v6_results) / n
t_avg = sum(r['dd']['chars'] for r in v6b_results) / n
g_clean = sum(1 for r in v6_results if not r['dd']['issues'] and r['dd']['chars'] > 0)
t_clean = sum(1 for r in v6b_results if not r['dd']['issues'] and r['dd']['chars'] > 0)

print(f"\n  v6 greedy:  avg {g_avg:.0f}ch, {g_clean}/{n} clean")
print(f"  v6b temp:   avg {t_avg:.0f}ch, {t_clean}/{n} clean")


# ============================================================
# SAVE
# ============================================================

all_results = {
    'config': {
        'model': model_name,
        'temperature': 1.15,
        'top_k': 5,
        'overlap_threshold': 0.40,
        'internal_rep_threshold': 0.30,
        'sat_threshold': 0.07,
        'branch_tokens': 64,
        'steps_per_branch': 28,
        'max_branches': 16,
        'max_total_tokens': 768,
    },
    'v6b_results': v6b_results,
    'v6_greedy_results': v6_results,
}

outpath = '/data/dendritic_v6b_results.json'
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
