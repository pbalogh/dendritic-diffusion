"""
Dendritic Diffusion: Depth-First Text Generation via Masked Language Modeling
=============================================================================

This script implements dendritic diffusion — a text generation algorithm where
output grows incrementally from a seed, like a crystal. Instead of filling a
fixed token window all at once (breadth-first), we grow short branches of text,
measure how much "pressure" the model still has to continue (supersaturation),
and keep growing as long as that signal is strong.

The model is LLaDA-8B-Base, a masked diffusion language model (MDLM). Unlike
autoregressive models (GPT-2, LLaMA) that generate one token at a time left
to right, LLaDA generates by iteratively "denoising" — starting from all-mask
tokens and progressively replacing masks with real tokens, most-confident first.

Key concepts:
  - MASKED DIFFUSION: Generation starts with [MASK] tokens. The model predicts
    what each mask should be. We reveal the most confident predictions first
    and re-run the model, iterating until all masks are filled.
  - SUPERSATURATION: After generating a branch of text, we append more masks
    and ask: "how confident is the model about what comes next?" High
    confidence = high supersaturation = keep growing. Low = stop.
  - STRUCTURAL vs SEMANTIC tokens: Function words (the, is, of) vs content
    words. Their ratio in the supersaturation probe tells us whether the model
    is predicting scaffolding (structural) or actual content (semantic).
  - DENDRITIC GROWTH: Like a crystal growing branch by branch, each new
    branch extends the text only as far as the model has something to say.
    The output length is determined by the model's confidence, not a fixed budget.
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# The special token that LLaDA uses to represent "unknown" positions.
# During generation, we fill a window with MASK_ID and let the model
# iteratively replace them with real tokens.
MASK_ID = 126336
model_name = 'GSAI-ML/LLaDA-8B-Base'

# ── Model Loading ──────────────────────────────────────────────────────
# We use 4-bit quantization (NF4) to fit the 8B-parameter model on a
# single GPU with ~15GB VRAM (e.g., NVIDIA T4). BitsAndBytes handles
# the quantization transparently — the model runs in float16 compute
# with 4-bit storage.

print(f"Loading {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Store weights in 4-bit
    bnb_4bit_compute_dtype=torch.float16, # Compute in float16
    bnb_4bit_quant_type="nf4",            # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants too
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/hf_cache',
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir='/data/hf_cache', quantization_config=bnb_config,
    device_map='auto',          # Automatically place layers across available GPUs
    trust_remote_code=True,     # LLaDA uses custom modeling code
).eval()                        # Inference mode — no gradient tracking
device = next(p.device for p in model.parameters() if p.device.type == 'cuda')

# ── Special Token Detection ────────────────────────────────────────────
# End-of-text tokens signal that the model wants to stop generating.
# We track these during denoising to detect when the model has "said
# everything it wants to say."
END_IDS = set()
for tok_str in ['<|eot_id|>', '<|end_of_text|>']:
    tid = tokenizer.convert_tokens_to_ids(tok_str)
    if tid is not None and tid != tokenizer.unk_token_id:
        END_IDS.add(tid)
if tokenizer.eos_token_id:
    END_IDS.add(tokenizer.eos_token_id)

# ── Structural Token Set ───────────────────────────────────────────────
# Function words, punctuation, and grammatical particles. When the model's
# supersaturation probe predicts mostly structural tokens, it means the
# model knows the *scaffolding* of what comes next (grammatical frame)
# but not the *content*. When it predicts semantic/content tokens, the
# model has a specific continuation in mind.
#
# This distinction matters: high structural + low semantic supersaturation
# means "I know a sentence is coming but not what it's about" — a weaker
# growth signal than high semantic supersaturation ("I know exactly what
# to say next").

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
    # Tokenizers may encode "the" and " the" (with leading space) differently,
    # so we add both variants to catch all forms.
    for variant in [word, ' ' + word]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        STRUCTURAL_TOKENS.update(ids)

print(f"Model loaded. {len(STRUCTURAL_TOKENS)} structural token ids, {len(END_IDS)} end ids.")


# ============================================================
# PROMPT FORMATTING
# ============================================================

def make_prompt_ids(seed_text):
    """
    Convert seed text to token IDs. We use the base model in raw text
    continuation mode — no chat template, no system prompt. The model
    will simply continue whatever text we provide.
    """
    return tokenizer.encode(seed_text, add_special_tokens=False)


# ============================================================
# CORE DENOISING — The Heart of Masked Diffusion
# ============================================================

def denoise_branch(input_ids, start, end, steps=28):
    """
    Iteratively denoise a span of MASK tokens into real text.

    How it works:
    1. The model sees the full sequence (prompt + previous branches + masks).
    2. For each mask position, it predicts a probability distribution over
       the vocabulary. We take the argmax (most likely token) and its
       confidence score.
    3. We sort all mask positions by confidence and reveal the top fraction.
       "Reveal" means replacing MASK_ID with the predicted token.
    4. Repeat with fewer masks each iteration, until all are filled.

    This is the core of masked diffusion: the model sees ALL positions
    simultaneously (unlike autoregressive models that only see the past),
    so it can use bidirectional context to make better predictions. The
    most confident positions are resolved first, creating a coarse-to-fine
    generation process.

    Args:
        input_ids: Full token sequence (list of ints, modified in place)
        start: Index of first MASK token in this branch
        end: Index after last MASK token
        steps: Number of denoising iterations (more = finer-grained reveal)

    Returns:
        input_ids: The sequence with masks replaced by predicted tokens
        step_log: Per-step diagnostics (EOS fraction, remaining masks)
    """
    step_log = []

    for step in range(steps):
        # Forward pass: model predicts logits for every position
        t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float()

        # Find remaining mask positions in this branch
        mask_positions = [i for i in range(start, end) if input_ids[i] == MASK_ID]
        if not mask_positions:
            break  # All masks filled — done

        # For each mask, get the model's top prediction and confidence
        cands = []
        n_eos = 0
        for i in mask_positions:
            probs = F.softmax(logits[i], dim=-1)
            p, idx = probs.max(dim=-1)
            if idx.item() in END_IDS:
                n_eos += 1
            cands.append((i, idx.item(), p.item()))

        # Track what fraction of masks want to be EOS — a signal that the
        # model wants to stop generating. With a base model this should be
        # low; with instruction-tuned models it tends to be very high.
        eos_frac = n_eos / max(1, len(mask_positions))
        step_log.append({
            'step': step,
            'eos_frac': round(eos_frac, 3),
            'n_masked': len(mask_positions),
        })

        # Unmask by confidence: reveal the most confident predictions first.
        # The number revealed per step increases as fewer masks remain,
        # creating an accelerating schedule (slow start, fast finish).
        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok

    return input_ids, step_log


# ============================================================
# SUPERSATURATION MEASUREMENT
# ============================================================

def measure_supersaturation(crystal, probe_tokens=48):
    """
    Measure how much "growth pressure" remains at the tip of the crystal.

    The metaphor comes from crystal growth: a supersaturated solution has
    more dissolved material than equilibrium allows, creating pressure to
    crystallize. Here, we ask: "if we appended more text, how confident
    would the model be about it?"

    Method: Append probe_tokens MASK tokens after the current text, run the
    model, and measure the confidence of the predictions for those probes.
    High average confidence = the model has a strong opinion about what
    comes next = high supersaturation = keep growing.

    We also decompose the signal into structural vs semantic components:
    - Structural supersaturation: confidence on function words/punctuation
    - Semantic supersaturation: confidence on content words
    This tells us whether the model knows the grammar of what comes next
    (structural) or the actual content (semantic).

    Args:
        crystal: Current token sequence (prompt + all generated branches)
        probe_tokens: How many mask tokens to append for measurement

    Returns:
        total_sat: Overall supersaturation (0-1, higher = more pressure)
        struct_sat: Fraction of probability on structural tokens
        semantic_sat: Fraction of probability on semantic tokens
        top_preds: The model's top predictions for the probe positions
        end_frac: Fraction of probes that predict EOS (exhaustion signal)
    """
    probe_ids = list(crystal) + [MASK_ID] * probe_tokens
    t = torch.tensor([probe_ids], device=device)

    with torch.no_grad():
        logits = model(t).logits[0].float()

    measurements = []
    for i in range(len(crystal), len(probe_ids)):
        probs = F.softmax(logits[i], dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        is_end = top_id.item() in END_IDS

        # Sum probability mass on structural (function word) tokens
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

    # Separate EOS predictions from content predictions
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

def _has_non_latin_script(text):
    """
    Return True if text contains any non-Latin script characters (CJK,
    Cyrillic, Arabic, etc.). Base models trained on multilingual data
    sometimes switch languages mid-generation — this catches that.
    """
    import unicodedata
    for c in text:
        if ord(c) > 0x024F:  # Beyond Latin Extended-B
            cat = unicodedata.category(c)
            if cat.startswith('L'):  # It's a letter, but non-Latin
                return True
    return False


def _cross_branch_repetitive(new_text, previous_texts, threshold=0.6):
    """
    Check if a new branch is too similar to any previous branch.

    Uses Jaccard similarity on trigrams: if more than 60% of the trigrams
    overlap with any single previous branch, the new branch is flagged as
    repetitive. This catches the degenerate mode where the model generates
    the same sentence over and over across branches, each passing internal
    quality checks but adding no new information.
    """
    if not previous_texts:
        return False
    new_words = new_text.lower().split()
    if len(new_words) < 4:
        return False
    new_grams = set(tuple(new_words[i:i+3]) for i in range(len(new_words) - 2))

    for prev in previous_texts:
        prev_words = prev.lower().split()
        if len(prev_words) < 4:
            continue
        prev_grams = set(tuple(prev_words[i:i+3]) for i in range(len(prev_words) - 2))
        intersection = len(new_grams & prev_grams)
        union = len(new_grams | prev_grams)
        if union > 0 and intersection / union > threshold:
            return True
    return False


def extract_sentences(tokens):
    """
    Clean raw denoised tokens into readable text.

    The denoising process fills all mask positions, but the result may
    include EOS tokens (model wanted to stop) or sentence fragments.
    This function:
    1. Truncates at the first EOS token
    2. Tries to end at a sentence boundary (period, question mark, etc.)
    3. Falls back to trimming trailing punctuation fragments
    """
    clean = []
    for t in tokens:
        if t in END_IDS:
            break
        clean.append(t)

    text = tokenizer.decode(clean).strip()

    # Try to end at a sentence boundary for cleaner output
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
# DENDRITIC GENERATION — The Main Algorithm
# ============================================================

def dendritic_generate(seed_text, branch_tokens=64, steps_per_branch=28,
                       max_branches=16, max_total_tokens=768):
    """
    Generate text by growing branches from a seed, guided by supersaturation.

    Algorithm:
    1. Start with the seed text (the "nucleus" of the crystal).
    2. Append branch_tokens MASK tokens and denoise them into text.
    3. Clean the result (extract complete sentences).
    4. Measure supersaturation: how confident is the model about continuing?
    5. If supersaturation is high enough, go to step 2 (grow another branch).
    6. If supersaturation drops below threshold, stop (crystal reached equilibrium).

    The result is a variable-length text whose size is determined by the model's
    confidence, not a fixed token budget. Comparative texts ("X vs Y") tend to
    produce many branches because each comparison point regenerates pressure.
    Narratives tend to produce few branches because stories have a single
    thread that exhausts quickly.

    Args:
        seed_text: The text to continue from
        branch_tokens: Number of MASK tokens per branch (shorter = more branches)
        steps_per_branch: Denoising iterations per branch (more = higher quality)
        max_branches: Safety limit on number of branches
        max_total_tokens: Safety limit on total generated tokens

    Returns:
        response: The full generated text (all branches concatenated)
        elapsed: Wall-clock time in seconds
        branches: Per-branch metadata (text, supersaturation, decision)
        all_step_logs: Per-step denoising diagnostics for each branch
    """
    prompt_ids = make_prompt_ids(seed_text)
    crystal = list(prompt_ids)
    plen = len(prompt_ids)
    branches = []
    branch_texts = []  # Track previous branch texts for cross-branch repetition detection
    total_tokens = 0
    consecutive_empty = 0
    all_step_logs = []

    t0 = time.time()

    for bi in range(max_branches):
        if total_tokens >= max_total_tokens:
            break

        # ── Grow a new branch ──
        # Append MASK tokens to the crystal and denoise them
        bstart = len(crystal)
        n = min(branch_tokens, max_total_tokens - total_tokens)
        ids = crystal + [MASK_ID] * n

        ids, step_log = denoise_branch(ids, bstart, len(ids), steps_per_branch)
        all_step_logs.append(step_log)

        # ── Extract clean text ──
        raw_branch = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw_branch)

        if not clean_text or len(clean_tokens) < 3:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break  # Two empty branches in a row = model is done
            continue

        consecutive_empty = 0

        # ── Cross-branch repetition check ──
        # Stop if this branch is too similar to any previous branch (Jaccard
        # similarity on trigrams > 0.6). This catches the degenerate mode where
        # the model confidently generates the same sentence across branches —
        # supersaturation stays high but no new information is being added.
        if _cross_branch_repetitive(clean_text, branch_texts):
            break

        # ── Language filter ──
        # Base models trained on multilingual data sometimes switch languages
        # mid-generation (e.g., inserting Chinese characters). Stop if the
        # branch contains any non-Latin script.
        if _has_non_latin_script(clean_text):
            break

        branch_texts.append(clean_text)
        crystal.extend(clean_tokens)
        total_tokens += len(clean_tokens)

        # ── Measure supersaturation ──
        # This is the key decision point: should we grow another branch?
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

        # ── Growth decision ──
        # The supersaturation thresholds determine when to stop:
        #   - Unanimous EOS + near-zero confidence → definite stop
        #   - High total supersaturation (>0.25) → definitely keep growing
        #   - Moderate supersaturation + high semantic ratio → keep going
        #     (the model has specific content in mind)
        #   - Moderate supersaturation + high structural ratio → keep going
        #     (the model knows the grammatical frame)
        #   - Below all thresholds → equilibrium reached, stop

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
    """
    Breadth-first baseline: fill ALL mask positions at once.

    This is the standard approach for masked diffusion — allocate a fixed
    window of max_tokens masks and denoise them all simultaneously. It's
    faster (one allocation, one denoising pass) but wastes compute on
    positions the model has nothing to say for, and can't adapt output
    length to content.
    """
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
    """Flag repetitions and other generation artifacts."""
    issues = []
    words = text.split()
    if len(words) < 5:
        return ['too short']

    # Adjacent word repetition ("the the")
    for i in range(len(words)-1):
        if words[i].lower() == words[i+1].lower() and words[i].lower() not in {
            'the','a','had','very','that','is','and'}:
            issues.append(f"repeated '{words[i]}'")

    # Trigram repetition (same 3-word phrase appearing 3+ times)
    trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words)-2)]
    seen = {}
    for tg in trigrams:
        seen[tg] = seen.get(tg, 0) + 1
    for tg, cnt in seen.items():
        if cnt >= 3 and tg not in {'of the the','in the the','the united states','as well as'}:
            issues.append(f"repeated '{tg}' ×{cnt}")

    return issues


def count_sentences(text):
    """Count sentences by splitting on terminal punctuation."""
    return len([s for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 5])


# ============================================================
# SEED TEXTS
# ============================================================
# These are text continuation prompts — the model simply continues
# whatever text we provide. Different discourse types produce different
# branching patterns (see README.md for the taxonomy results).

seeds = [
    # Expository — factual explanation, tends to branch into subtopics
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. It can be broadly divided into two main components:",

    # Narrative — story continuation, tends to stay linear (low branching)
    "On July 16, 1969, the Saturn V rocket carrying the Apollo 11 crew lifted off from Kennedy Space Center. Commander Neil Armstrong, Lunar Module Pilot Buzz Aldrin, and Command Module Pilot Michael Collins",

    # Causal — cause-and-effect chains
    "The French Revolution of 1789 was the product of decades of social inequality, economic crisis, and political dysfunction. The causes can be traced to several interrelated factors:",

    # Expository (scientific) — mechanistic explanation
    "Evolution by natural selection, first proposed by Charles Darwin and Alfred Russel Wallace, is the process by which populations of organisms change over successive generations. The mechanism works through several key principles:",

    # Comparative — "X vs Y" structure, tends to produce the most branches
    # because alternating between comparison points regenerates supersaturation
    "Throughout history, human societies have organized themselves under various forms of government. Three of the most significant are democracy, monarchy, and authoritarianism, each with distinct characteristics:",

    # Expository (contemporary) — topic with many subtopics
    "Climate change refers to the long-term shift in global temperatures and weather patterns. While climate has changed naturally throughout Earth's history, the current rapid warming is primarily driven by human activities, particularly",
]

seed_labels = [
    "Immune System",
    "Apollo 11",
    "French Revolution",
    "Evolution",
    "Government Types",
    "Climate Change",
]


# ============================================================
# RUN: Compare Dendritic vs Breadth-First Generation
# ============================================================

print(f"\n{'=' * 75}")
print("DENDRITIC DIFFUSION: LLaDA-8B-Base")
print("Comparing depth-first (dendritic) vs breadth-first generation")
print(f"{'=' * 75}")

results = []

for seed_text, label in zip(seeds, seed_labels):
    print(f"\n{'═' * 75}")
    print(f"SEED [{label}]: \"{seed_text[:70]}...\"")
    print(f"{'═' * 75}")

    # ── Breadth-first baseline ──
    bf_text, bf_time, bf_log = breadth_first(seed_text, max_tokens=768, steps=64)
    bf_issues = check_quality(bf_text)
    bf_sents = count_sentences(bf_text)

    # ── Dendritic generation ──
    dd_text, dd_time, dd_branches, dd_logs = dendritic_generate(
        seed_text, branch_tokens=64, steps_per_branch=28,
        max_branches=16, max_total_tokens=768,
    )
    dd_issues = check_quality(dd_text)
    dd_sents = count_sentences(dd_text)

    # ── Display breadth-first result ──
    print(f"\n  BREADTH-FIRST ({bf_sents} sentences, {len(bf_text)} chars, {bf_time:.1f}s):")
    for line in bf_text[:500].split('\n'):
        print(f"    {line}")
    if len(bf_text) > 500:
        print(f"    [... +{len(bf_text)-500} chars]")
    if bf_issues:
        print(f"    ⚠️  {len(bf_issues)} issues: {', '.join(bf_issues[:3])}")
    else:
        print(f"    ✅ Clean")

    # EOS diagnostic: what fraction of initial masks predicted EOS?
    if bf_log:
        eos_fracs = [e['eos_frac'] for e in bf_log[:5]]
        print(f"    📊 EOS at steps 0-4: {[f'{f:.0%}' for f in eos_fracs]}")

    # ── Display dendritic result ──
    print(f"\n  DENDRITIC ({dd_sents} sentences, {len(dd_branches)} branches, {len(dd_text)} chars, {dd_time:.1f}s):")
    for line in dd_text[:800].split('\n'):
        print(f"    {line}")
    if len(dd_text) > 800:
        print(f"    [... +{len(dd_text)-800} chars]")
    if dd_issues:
        print(f"    ⚠️  {len(dd_issues)} issues: {', '.join(dd_issues[:5])}")
    else:
        print(f"    ✅ Clean")

    # ── Crystal growth log: the supersaturation trace ──
    print(f"\n  📊 Crystal growth (branch → supersaturation → decision):")
    for b in dd_branches:
        sat = b['supersaturation']
        print(f"    B{b['idx']}: sat={sat['total']:.3f} "
              f"[struct:{sat['structural']:.2f} / sem:{sat['semantic']:.2f}] "
              f"end={sat['end_fraction']:.0%} → {b['decision'][:55]}")
        print(f"         \"{b['text'][:80]}{'...' if len(b['text'])>80 else ''}\"")

    # Step-by-step EOS for first branch
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
print("SUMMARY")
print(f"{'=' * 75}")

n = len(seeds)

print(f"\n  {'Seed':<25s} {'BF chars':>10s} {'DD chars':>10s} {'Branches':>10s} {'DD sents':>10s} {'Quality':>10s}")
print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for r in results:
    q = '✅' if not r['dd']['issues'] else f"⚠️{len(r['dd']['issues'])}"
    print(f"  {r['label']:<25s} {r['bf']['chars']:>10d} {r['dd']['chars']:>10d} "
          f"{len(r['dd']['branches']):>10d} {r['dd']['sents']:>10d} {q:>10s}")

bf_avg = sum(r['bf']['chars'] for r in results) / n
dd_avg = sum(r['dd']['chars'] for r in results) / n
dd_branches_avg = sum(len(r['dd']['branches']) for r in results) / n
dd_sents_avg = sum(r['dd']['sents'] for r in results) / n
bf_sents_avg = sum(r['bf']['sents'] for r in results) / n
dd_clean = sum(1 for r in results if not r['dd']['issues'] and r['dd']['chars'] > 0)
bf_clean = sum(1 for r in results if not r['bf']['issues'] and r['bf']['chars'] > 0)

print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
print(f"  {'AVERAGE':<25s} {bf_avg:>10.0f} {dd_avg:>10.0f} {dd_branches_avg:>10.1f} {dd_sents_avg:>10.1f}")

print(f"\n  Breadth-first: {bf_clean}/{n} clean, avg {bf_avg:.0f} chars, {bf_sents_avg:.1f} sentences")
print(f"  Dendritic:     {dd_clean}/{n} clean, avg {dd_avg:.0f} chars, {dd_sents_avg:.1f} sentences, {dd_branches_avg:.1f} branches")


# ============================================================
# SAVE RESULTS
# ============================================================

all_results = {
    'config': {
        'model': model_name,
        'prompt_style': 'text continuation (base model)',
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
