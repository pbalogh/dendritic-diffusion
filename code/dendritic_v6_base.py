"""
Dendritic Diffusion v6: Base Model — No More Fighting EOS
==========================================================
v5e diagnosed and treated the symptom: LLaDA-8B-Instruct's SFT training
creates a powerful EOS attractor that kills nucleation. Temperature annealing
+ EOS penalties are hacks that fight the model's training signal.

v6 eliminates the disease: switch to LLaDA-8B-Base, which was trained on
raw text completion. No SFT, no EOS bias, no chat template. The model
WANTS to continue text — that's what pretraining teaches.

Key changes from v5e:
  1. Model: LLaDA-8B-Base (not Instruct)
  2. Prompt: text continuation seeds (not chat template + system prompt)
  3. No temperature annealing, no EOS penalty — unnecessary with Base
  4. Clean greedy denoising throughout — the model cooperates now
  5. Same supersaturation stopping criterion
  6. Same crystal growth / branch architecture

The philosophical alignment is better too: we're seeding a crystallization
process, not asking a chatbot a question. Text continuation IS crystallization.
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time
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

print(f"Model loaded. {len(STRUCTURAL_TOKENS)} structural token ids, {len(END_IDS)} end ids.")


# ============================================================
# PROMPT FORMATTING — text continuation, not chat
# ============================================================

def make_prompt_ids(seed_text):
    """
    Base model prompt: just raw text. The model will continue it.
    No chat template, no system prompt, no special tokens.
    """
    return tokenizer.encode(seed_text, add_special_tokens=False)


# ============================================================
# CORE DENOISING — clean greedy, no hacks needed
# ============================================================

def denoise_branch(input_ids, start, end, steps=28):
    """
    Greedy masked denoising. With Base model, this should work cleanly
    without temperature annealing or EOS penalties.
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
            probs = F.softmax(logits[i], dim=-1)
            p, idx = probs.max(dim=-1)
            if idx.item() in END_IDS:
                n_eos += 1
            cands.append((i, idx.item(), p.item()))
        
        eos_frac = n_eos / max(1, len(mask_positions))
        step_log.append({
            'step': step,
            'eos_frac': round(eos_frac, 3),
            'n_masked': len(mask_positions),
        })
        
        # Unmask by confidence (most confident first)
        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok
    
    return input_ids, step_log


# ============================================================
# SUPERSATURATION MEASUREMENT
# ============================================================

def measure_supersaturation(crystal, probe_tokens=48):
    """Probe what the model predicts beyond current crystal."""
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


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_sentences(tokens):
    """Extract clean sentences from generated tokens."""
    clean = []
    for t in tokens:
        if t in END_IDS:
            break
        clean.append(t)
    
    text = tokenizer.decode(clean).strip()
    
    # Try to end at sentence boundary
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
# DENDRITIC GENERATION
# ============================================================

def dendritic_generate(seed_text, branch_tokens=64, steps_per_branch=28,
                       max_branches=16, max_total_tokens=768):
    """
    Dendritic generation with Base model.
    
    Higher defaults than v5e (max_branches=16, max_tokens=768) because
    we expect the Base model to actually produce content without fighting us.
    """
    prompt_ids = make_prompt_ids(seed_text)
    crystal = list(prompt_ids)
    plen = len(prompt_ids)
    branches = []
    total_tokens = 0
    consecutive_empty = 0
    all_step_logs = []
    
    t0 = time.time()
    
    for bi in range(max_branches):
        if total_tokens >= max_total_tokens:
            break
        
        bstart = len(crystal)
        n = min(branch_tokens, max_total_tokens - total_tokens)
        ids = crystal + [MASK_ID] * n
        
        ids, step_log = denoise_branch(ids, bstart, len(ids), steps_per_branch)
        all_step_logs.append(step_log)
        
        raw_branch = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw_branch)
        
        if not clean_text or len(clean_tokens) < 3:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue
        
        consecutive_empty = 0
        crystal.extend(clean_tokens)
        total_tokens += len(clean_tokens)
        
        total_sat, struct_sat, semantic_sat, top_preds, end_frac = \
            measure_supersaturation(crystal)
        
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
        }
        
        # Supersaturation-based stopping
        if total_sat < 0.05 and end_frac >= 0.99:
            branch['decision'] = f'STOP (zero energy, unanimous EOS)'
            branches.append(branch)
            break
        
        if total_sat >= 0.25:
            branch['decision'] = f'GROW (sat={total_sat:.3f} — high energy)'
            branches.append(branch)
            continue
        
        if total_sat >= 0.10 and semantic_sat > 0.55:
            branch['decision'] = f'GROW (sat={total_sat:.3f}, sem={semantic_sat:.2f})'
            branches.append(branch)
            continue
        
        if total_sat >= 0.10 and struct_sat > 0.45:
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


def breadth_first(seed_text, max_tokens=768, steps=64):
    """BF baseline — also using Base model with text continuation."""
    prompt_ids = make_prompt_ids(seed_text)
    plen = len(prompt_ids)
    ids = prompt_ids + [MASK_ID] * max_tokens
    
    t0 = time.time()
    ids, step_log = denoise_branch(ids, plen, len(ids), steps)
    elapsed = time.time() - t0
    
    resp = []
    for t in ids[plen:]:
        if t in END_IDS:
            break
        resp.append(t)
    
    return tokenizer.decode(resp).strip(), elapsed, step_log


# ============================================================
# QUALITY CHECKS
# ============================================================

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
            issues.append(f"repeated '{tg}' ×{cnt}")
    
    return issues


def count_sentences(text):
    return len([s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 5])


# ============================================================
# TEXT CONTINUATION SEEDS (not questions!)
# ============================================================
# These seed a crystallization process. The model continues the text,
# not answers a question. This is the natural mode for a base model.

seeds = [
    # Expository — should produce long detailed text naturally
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. It can be broadly divided into two main components:",
    
    # Narrative — story continuation
    "On July 16, 1969, the Saturn V rocket carrying the Apollo 11 crew lifted off from Kennedy Space Center. Commander Neil Armstrong, Lunar Module Pilot Buzz Aldrin, and Command Module Pilot Michael Collins",
    
    # Analytical — causes and effects
    "The French Revolution of 1789 was the product of decades of social inequality, economic crisis, and political dysfunction. The causes can be traced to several interrelated factors:",
    
    # Scientific explanation
    "Evolution by natural selection, first proposed by Charles Darwin and Alfred Russel Wallace, is the process by which populations of organisms change over successive generations. The mechanism works through several key principles:",
    
    # Comparative analysis
    "Throughout history, human societies have organized themselves under various forms of government. Three of the most significant are democracy, monarchy, and authoritarianism, each with distinct characteristics:",
    
    # Current topic with detail
    "Climate change refers to the long-term shift in global temperatures and weather patterns. While climate has changed naturally throughout Earth's history, the current rapid warming is primarily driven by human activities, particularly",
]

# Short labels for display
seed_labels = [
    "Immune System",
    "Apollo 11",
    "French Revolution",
    "Evolution",
    "Government Types",
    "Climate Change",
]


# ============================================================
# RUN: v6 Base Model
# ============================================================

print(f"\n{'=' * 75}")
print("DENDRITIC DIFFUSION v6: LLaDA-8B-Base (text continuation, no annealing)")
print(f"{'=' * 75}")

results = []

for seed_text, label in zip(seeds, seed_labels):
    print(f"\n{'═' * 75}")
    print(f"SEED [{label}]: \"{seed_text[:70]}...\"")
    print(f"{'═' * 75}")
    
    # Breadth-first baseline
    bf_text, bf_time, bf_log = breadth_first(seed_text, max_tokens=768, steps=64)
    bf_issues = check_quality(bf_text)
    bf_sents = count_sentences(bf_text)
    
    # Dendritic
    dd_text, dd_time, dd_branches, dd_logs = dendritic_generate(
        seed_text, branch_tokens=64, steps_per_branch=28,
        max_branches=16, max_total_tokens=768,
    )
    dd_issues = check_quality(dd_text)
    dd_sents = count_sentences(dd_text)
    
    # Display BF
    print(f"\n  BF ({bf_sents} sent, {len(bf_text)} chars, {bf_time:.1f}s):")
    for line in bf_text[:500].split('\n'):
        print(f"    {line}")
    if len(bf_text) > 500:
        print(f"    [... +{len(bf_text)-500} chars]")
    if bf_issues:
        print(f"    ⚠️  {len(bf_issues)} issues: {', '.join(bf_issues[:3])}")
    else:
        print(f"    ✅ Clean")
    
    # Show BF EOS fractions (key diagnostic — should be LOW with Base model)
    if bf_log:
        eos_fracs = [e['eos_frac'] for e in bf_log[:5]]
        print(f"    📊 EOS at steps 0-4: {[f'{f:.0%}' for f in eos_fracs]}")
    
    # Display DD
    print(f"\n  DD ({dd_sents} sent, {len(dd_branches)} branches, {len(dd_text)} chars, {dd_time:.1f}s):")
    for line in dd_text[:800].split('\n'):
        print(f"    {line}")
    if len(dd_text) > 800:
        print(f"    [... +{len(dd_text)-800} chars]")
    if dd_issues:
        print(f"    ⚠️  {len(dd_issues)} issues: {', '.join(dd_issues[:5])}")
    else:
        print(f"    ✅ Clean")
    
    # Crystal growth log
    print(f"\n  📊 Crystal growth:")
    for b in dd_branches:
        sat = b['supersaturation']
        print(f"    B{b['idx']}: sat={sat['total']:.3f} "
              f"[S:{sat['structural']:.2f}/C:{sat['semantic']:.2f}] "
              f"end={sat['end_fraction']:.0%} → {b['decision'][:55]}")
        print(f"         \"{b['text'][:80]}{'...' if len(b['text'])>80 else ''}\"")
    
    # Step-by-step EOS for first branch (the diagnostic that matters)
    if dd_logs and dd_logs[0]:
        eos_trace = [e['eos_frac'] for e in dd_logs[0][:8]]
        print(f"\n  🔬 Branch 0 EOS trace: {[f'{f:.0%}' for f in eos_trace]}")
    
    results.append({
        'label': label,
        'seed': seed_text,
        'bf': {'text': bf_text, 'time': bf_time, 'sents': bf_sents,
               'issues': bf_issues, 'chars': len(bf_text),
               'step_log': bf_log},
        'dd': {'text': dd_text, 'time': dd_time, 'sents': dd_sents,
               'issues': dd_issues, 'chars': len(dd_text),
               'branches': dd_branches, 'step_logs': dd_logs},
    })


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'=' * 75}")
print("SUMMARY: v6 Base Model Results")
print(f"{'=' * 75}")

n = len(seeds)

print(f"\n  {'Seed':<25s} {'BF chars':>10s} {'DD chars':>10s} {'DD branches':>12s} {'DD sents':>10s} {'Quality':>10s}")
print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

for r in results:
    bf_chars = r['bf']['chars']
    dd_chars = r['dd']['chars']
    dd_b = len(r['dd']['branches'])
    dd_s = r['dd']['sents']
    q = '✅' if not r['dd']['issues'] else f"⚠️{len(r['dd']['issues'])}"
    print(f"  {r['label']:<25s} {bf_chars:>10d} {dd_chars:>10d} {dd_b:>12d} {dd_s:>10d} {q:>10s}")

# Averages
bf_avg = sum(r['bf']['chars'] for r in results) / n
dd_avg = sum(r['dd']['chars'] for r in results) / n
dd_branches_avg = sum(len(r['dd']['branches']) for r in results) / n
dd_sents_avg = sum(r['dd']['sents'] for r in results) / n
bf_sents_avg = sum(r['bf']['sents'] for r in results) / n
dd_clean = sum(1 for r in results if not r['dd']['issues'] and r['dd']['chars'] > 0)
bf_clean = sum(1 for r in results if not r['bf']['issues'] and r['bf']['chars'] > 0)

print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")
print(f"  {'AVERAGE':<25s} {bf_avg:>10.0f} {dd_avg:>10.0f} {dd_branches_avg:>12.1f} {dd_sents_avg:>10.1f}")

print(f"\n  Key metrics:")
print(f"    BF: {bf_clean}/{n} clean, avg {bf_avg:.0f} chars, {bf_sents_avg:.1f} sents")
print(f"    DD: {dd_clean}/{n} clean, avg {dd_avg:.0f} chars, {dd_sents_avg:.1f} sents, {dd_branches_avg:.1f} branches")

# The key diagnostic: EOS fraction at step 0
print(f"\n  🔬 EOS fraction at step 0 (should be LOW with Base model):")
for r in results:
    bf_eos0 = r['bf']['step_log'][0]['eos_frac'] if r['bf']['step_log'] else '?'
    dd_eos0 = r['dd']['step_logs'][0][0]['eos_frac'] if r['dd']['step_logs'] and r['dd']['step_logs'][0] else '?'
    print(f"    {r['label']:<25s} BF={bf_eos0}  DD={dd_eos0}")


# ============================================================
# SAVE
# ============================================================

all_results = {
    'config': {
        'model': model_name,
        'model_type': 'base (no SFT)',
        'prompt_style': 'text continuation',
        'annealing': False,
        'eos_penalty': None,
        'branch_tokens': 64,
        'steps_per_branch': 28,
        'max_branches': 16,
        'max_total_tokens': 768,
        'bf_max_tokens': 768,
        'bf_steps': 64,
    },
    'results': results,
    'summary': {
        'bf_avg_chars': bf_avg,
        'dd_avg_chars': dd_avg,
        'dd_avg_branches': dd_branches_avg,
        'dd_avg_sents': dd_sents_avg,
        'bf_avg_sents': bf_sents_avg,
        'dd_clean': dd_clean,
        'bf_clean': bf_clean,
        'n_prompts': n,
    },
}

outpath = '/data/dendritic_v6_base_results.json'
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {outpath}")
print(f"{'=' * 75}")
