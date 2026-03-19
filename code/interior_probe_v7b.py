"""
Interior Probing v7b: Iterative Mullins-Sekerka Branching
==========================================================
v7 proved interior supersaturation is 2-8x higher than tip sat across
all seeds. Text WANTS to branch internally, not just extend.

v7b implements the full loop:
  1. Grow initial crystal (linear, modest)
  2. Probe ALL interior sentence boundaries
  3. Branch at highest-energy point (insert elaboration)
  4. Re-probe the updated crystal
  5. Repeat until all interior points are at equilibrium
  6. Compare final tree-structured text with linear-only text

This is true tree-structured generation driven by physics.
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

print(f"Model loaded.")


def make_prompt_ids(seed_text):
    return tokenizer.encode(seed_text, add_special_tokens=False)


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


def find_sentence_boundaries(token_ids, prompt_len):
    """Find sentence boundary positions in the crystal."""
    boundaries = []
    for i, tid in enumerate(token_ids[prompt_len:], start=prompt_len):
        tok_text = tokenizer.decode([tid])
        if any(c in tok_text for c in '.!?') and i > prompt_len + 5:
            boundaries.append(i + 1)
    return boundaries


def probe_single_point(crystal_ids, position, probe_tokens=24):
    """Measure supersaturation at a single interior point."""
    probed_ids = list(crystal_ids[:position]) + [MASK_ID] * probe_tokens + list(crystal_ids[position:])
    
    if len(probed_ids) > 2048:
        return None
    
    t = torch.tensor([probed_ids], device=device)
    with torch.no_grad():
        logits = model(t).logits[0].float()
    
    confs = []
    n_end = 0
    top_preds = []
    for i in range(position, position + probe_tokens):
        probs = F.softmax(logits[i], dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        if top_id.item() in END_IDS:
            n_end += 1
        else:
            confs.append(top_prob.item())
            if len(top_preds) < 4:
                top_preds.append(tokenizer.decode([top_id.item()]).strip())
    
    if not confs:
        return {'sat': 0.0, 'end_frac': 1.0, 'top_preds': []}
    
    return {
        'sat': sum(confs) / len(confs),
        'end_frac': n_end / probe_tokens,
        'top_preds': top_preds,
    }


def probe_all_interior(crystal_ids, prompt_len, probe_tokens=24):
    """Probe supersaturation at all interior sentence boundaries."""
    boundaries = find_sentence_boundaries(crystal_ids, prompt_len)
    probes = []
    
    for bp in boundaries:
        if bp >= len(crystal_ids) - 5:
            continue
        
        result = probe_single_point(crystal_ids, bp, probe_tokens)
        if result is None:
            continue
        
        context_before = tokenizer.decode(crystal_ids[max(prompt_len, bp-20):bp]).strip()[-50:]
        context_after = tokenizer.decode(crystal_ids[bp:min(len(crystal_ids), bp+20)]).strip()[:50]
        
        probes.append({
            'position': bp,
            'offset': bp - prompt_len,
            'sat': round(result['sat'], 4),
            'end_frac': round(result['end_frac'], 3),
            'top_preds': result['top_preds'],
            'context_before': context_before,
            'context_after': context_after,
        })
    
    return probes


def grow_side_branch_and_insert(crystal_ids, insert_pos, branch_tokens=48, steps=28):
    """
    Grow a side branch at an interior position and return the new crystal.
    Insert MASKs, denoise them, clean up, return updated crystal.
    """
    ids = list(crystal_ids[:insert_pos]) + [MASK_ID] * branch_tokens + list(crystal_ids[insert_pos:])
    
    if len(ids) > 2048:
        # Trim from end
        ids = ids[:2048]
    
    ids = denoise_branch(ids, insert_pos, insert_pos + branch_tokens, steps, 1.15)
    
    # Extract the generated branch text
    branch_raw = ids[insert_pos:insert_pos + branch_tokens]
    clean_tokens, clean_text = extract_sentences(branch_raw)
    
    if not clean_text or len(clean_tokens) < 3:
        return None, "", crystal_ids
    
    # Check for repetition
    if internal_repetition(clean_text) > 0.30:
        return None, f"[REPETITIVE: {clean_text[:50]}]", crystal_ids
    
    # Check if branch is just commas/periods (degenerate)
    alpha_ratio = sum(1 for c in clean_text if c.isalpha()) / max(1, len(clean_text))
    if alpha_ratio < 0.3:
        return None, f"[DEGENERATE: {clean_text[:50]}]", crystal_ids
    
    # Insert clean branch into crystal
    new_crystal = list(crystal_ids[:insert_pos]) + clean_tokens + list(crystal_ids[insert_pos:])
    
    return clean_tokens, clean_text, new_crystal


def grow_initial_crystal(seed_text, max_branches=5, max_tokens=256):
    """Grow a modest initial crystal (linear)."""
    prompt_ids = make_prompt_ids(seed_text)
    crystal = list(prompt_ids)
    plen = len(prompt_ids)
    total_tokens = 0
    consecutive_empty = 0
    
    for bi in range(max_branches):
        if total_tokens >= max_tokens:
            break
        
        bstart = len(crystal)
        n = min(64, max_tokens - total_tokens)
        ids = crystal + [MASK_ID] * n
        ids = denoise_branch(ids, bstart, len(ids), 28, 1.15)
        
        raw = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw)
        
        if not clean_text or len(clean_tokens) < 3:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        
        consecutive_empty = 0
        if internal_repetition(clean_text) > 0.30:
            continue
        
        alpha_ratio = sum(1 for c in clean_text if c.isalpha()) / max(1, len(clean_text))
        if alpha_ratio < 0.3:
            continue
        
        crystal.extend(clean_tokens)
        total_tokens += len(clean_tokens)
    
    return crystal, plen


# ============================================================
# ITERATIVE INTERIOR BRANCHING
# ============================================================

def iterative_interior_branching(seed_text, max_iterations=6,
                                  branch_threshold=0.15,
                                  max_total_tokens=768):
    """
    The full iterative loop:
    1. Grow initial crystal
    2. Probe interior
    3. Branch at highest-energy point
    4. Re-probe
    5. Repeat until equilibrium or budget
    """
    t0 = time.time()
    
    # Phase 1: Initial crystal
    crystal, plen = grow_initial_crystal(seed_text, max_branches=5, max_tokens=256)
    initial_text = tokenizer.decode(crystal[plen:]).strip()
    initial_tokens = len(crystal) - plen
    
    print(f"  Initial crystal: {initial_tokens} tokens")
    print(f"  \"{initial_text[:200]}...\"")
    
    iterations = []
    
    for it in range(max_iterations):
        if len(crystal) - plen >= max_total_tokens:
            print(f"\n  Iteration {it}: token budget reached ({len(crystal)-plen}/{max_total_tokens})")
            break
        
        # Probe interior
        probes = probe_all_interior(crystal, plen, probe_tokens=24)
        
        if not probes:
            print(f"\n  Iteration {it}: no interior points to probe")
            break
        
        # Sort by supersaturation
        probes.sort(key=lambda p: -p['sat'])
        best = probes[0]
        
        print(f"\n  Iteration {it}: max interior sat={best['sat']:.3f} at offset {best['offset']}")
        print(f"    ...{best['context_before'][-40:]}")
        print(f"    → predictions: {best['top_preds'][:4]}")
        print(f"    {best['context_after'][:40]}...")
        
        # Check threshold
        if best['sat'] < branch_threshold:
            print(f"    ❄️  Below threshold ({branch_threshold}) — crystal at equilibrium")
            iterations.append({
                'iteration': it,
                'max_sat': best['sat'],
                'action': 'EQUILIBRIUM',
            })
            break
        
        # Branch!
        print(f"    🌿 Branching at offset {best['offset']}...")
        branch_toks, branch_text, new_crystal = grow_side_branch_and_insert(
            crystal, plen + best['offset'],
            branch_tokens=48, steps=28,
        )
        
        if branch_toks is None:
            print(f"    ⚠️  Branch failed: {branch_text}")
            iterations.append({
                'iteration': it,
                'max_sat': best['sat'],
                'action': f'BRANCH_FAILED ({branch_text[:50]})',
                'offset': best['offset'],
            })
            # Try next-highest point? For now just skip
            continue
        
        crystal = new_crystal
        added = len(branch_toks)
        
        print(f"    ✅ Inserted {added} tokens: \"{branch_text[:100]}...\"")
        
        iterations.append({
            'iteration': it,
            'max_sat': best['sat'],
            'action': 'BRANCHED',
            'offset': best['offset'],
            'branch_text': branch_text,
            'branch_tokens': added,
            'total_tokens': len(crystal) - plen,
        })
    
    elapsed = time.time() - t0
    final_text = tokenizer.decode(crystal[plen:]).strip()
    
    return {
        'initial_text': initial_text,
        'initial_tokens': initial_tokens,
        'final_text': final_text,
        'final_tokens': len(crystal) - plen,
        'iterations': iterations,
        'n_successful_branches': sum(1 for it in iterations if it['action'] == 'BRANCHED'),
        'elapsed': round(elapsed, 1),
    }


# ============================================================
# RUN
# ============================================================

seeds = [
    ("Immune System",
     "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. It can be broadly divided into two main components:"),
    ("French Revolution",
     "The French Revolution of 1789 was the product of decades of social inequality, economic crisis, and political dysfunction. The causes can be traced to several interrelated factors:"),
    ("Climate Change",
     "Climate change refers to the long-term shift in global temperatures and weather patterns. While climate has changed naturally throughout Earth's history, the current rapid warming is primarily driven by human activities, particularly"),
]

print(f"\n{'=' * 75}")
print("v7b: ITERATIVE INTERIOR BRANCHING")
print("Grow crystal → probe → branch → re-probe → repeat until equilibrium")
print(f"Branch threshold: sat > 0.15")
print(f"{'=' * 75}")

all_results = []

for label, seed_text in seeds:
    print(f"\n{'═' * 75}")
    print(f"SEED [{label}]")
    print(f"{'═' * 75}")
    
    result = iterative_interior_branching(
        seed_text,
        max_iterations=6,
        branch_threshold=0.15,
        max_total_tokens=768,
    )
    result['label'] = label
    result['seed'] = seed_text
    all_results.append(result)
    
    print(f"\n  ── Result ──")
    print(f"  Initial: {result['initial_tokens']} tokens")
    print(f"  Final:   {result['final_tokens']} tokens (+{result['final_tokens']-result['initial_tokens']} from {result['n_successful_branches']} branches)")
    print(f"  Time:    {result['elapsed']}s")
    print(f"\n  Final text:")
    # Show with branch markers
    lines = result['final_text'].split('\n')
    for line in lines[:15]:
        print(f"    {line[:100]}")
    if len(lines) > 15:
        print(f"    [...{len(lines)-15} more lines]")


# ============================================================
# COMPARISON: Linear vs Tree-structured
# ============================================================

print(f"\n{'=' * 75}")
print("COMPARISON: Linear (initial) vs Tree-structured (after interior branching)")
print(f"{'=' * 75}")

for r in all_results:
    print(f"\n  {r['label']}:")
    print(f"    Linear:  {r['initial_tokens']} tokens")
    print(f"    Tree:    {r['final_tokens']} tokens ({r['n_successful_branches']} interior branches)")
    print(f"    Growth:  +{r['final_tokens']-r['initial_tokens']} tokens ({(r['final_tokens']/max(1,r['initial_tokens'])-1)*100:.0f}% expansion)")
    
    # Check quality
    int_rep_initial = internal_repetition(r['initial_text'])
    int_rep_final = internal_repetition(r['final_text'])
    alpha_initial = sum(1 for c in r['initial_text'] if c.isalpha()) / max(1, len(r['initial_text']))
    alpha_final = sum(1 for c in r['final_text'] if c.isalpha()) / max(1, len(r['final_text']))
    
    print(f"    Repetition: {int_rep_initial:.0%} → {int_rep_final:.0%}")
    print(f"    Alpha ratio: {alpha_initial:.0%} → {alpha_final:.0%}")
    
    # Show iteration log
    for it in r['iterations']:
        action = it['action']
        sat = it.get('max_sat', 0)
        if action == 'BRANCHED':
            print(f"    ├─ iter {it['iteration']}: sat={sat:.3f} → 🌿 +{it['branch_tokens']}tok at offset {it['offset']}")
        elif action == 'EQUILIBRIUM':
            print(f"    └─ iter {it['iteration']}: sat={sat:.3f} → ❄️  equilibrium")
        else:
            print(f"    ├─ iter {it['iteration']}: sat={sat:.3f} → ⚠️  {action[:50]}")


# Save
outpath = '/data/interior_probe_v7b_results.json'
with open(outpath, 'w') as f:
    json.dump({
        'experiment': 'iterative_interior_branching',
        'version': 'v7b',
        'model': model_name,
        'branch_threshold': 0.15,
        'max_iterations': 6,
        'results': all_results,
    }, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
