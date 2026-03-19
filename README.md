# Dendritic Diffusion: L-System Grammars for Language Generation

**Depth-first text generation via masked diffusion, where text grows like a crystal.**

Dendritic Diffusion uses [LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) (a masked diffusion language model) with a novel generation algorithm inspired by L-system grammars and crystal growth. Instead of filling a fixed token window left-to-right, text branches outward from a seed — each branch grows only as long as the model has something confident to say, measured by **supersaturation** (an entropy-based continuation signal).

## Key Ideas

1. **Supersaturation-guided generation**: After each branch of text is denoised, we measure how much "pressure" remains — how confident the model is about what comes next. If supersaturation exceeds a threshold, we grow another branch. If not, we stop. The crystal determines its own size.

2. **L-system formalism**: Autoregressive generation = sequential rewriting (Chomsky). Masked diffusion = parallel rewriting (Lindenmayer). 60 years of L-system theory — bracketed systems, context-sensitive grammars, stochastic production rules — imports directly into diffusion language modeling.

3. **Discourse-dependent topology**: Different types of text produce different branching patterns. Comparative texts ("X vs Y") sustain supersaturation across alternating subtopics, producing rich tree structures. Narratives stay linear — stories resist branching.

## Examples

### Comparative: Arteries vs Veins → 20 branches, rising supersaturation

**Seed:** *"Arteries and veins are both blood vessels, but they differ in structure, function, and the direction of blood flow."*

**Output (588 tokens, 20 branches):**

> Arteries carry blood away from the heart, while veins bring blood back to the heart.
>
> 1\. Structure: Arteries have thicker walls than veins, which helps them to withstand the pressure of the heart pumping. They also have smooth muscle layers that help to regulate blood flow. Veins have thinner walls and fewer muscle layers, which allows them to return blood back to the heart.
>
> 2\. Function: Arteries carry blood away from the heart to the tissues and organs of the body. They deliver oxygen and nutrients to the cells...

**Supersaturation trace** (entropy-based, 0 = exhausted, 1 = maximum pressure):
```
0.57 → 0.56 → 0.60 → 0.67 → 0.65 → 0.61 → 0.54 → 0.56 → 0.64 → 0.58 → 0.57 → ...
                 ↑ oscillating — each new comparison point regenerates pressure
```

### Expository: Nervous System → 10 branches, oscillating

**Seed:** *"The nervous system is a highly complex part of an animal that coordinates its actions and sensory information by transmitting signals to and from different parts of its body."*

**Output (290 tokens, 10 branches):**

> In vertebrates, the nervous system consists of the central nervous system (CNS), which includes the brain and spinal cord, and the peripheral nervous system (PNS), which includes the peripheral nerves and ganglia. The brain is the control center of the nervous system and regulates various functions such as movement. The spinal cord is the main connection between the brain and the rest of the body, transmitting signals...

**Supersaturation trace:**
```
0.56 → 0.53 → 0.64 → 0.63 → 0.56 → 0.61 → 0.68 → 0.75 → 0.72 → 0.73
              ↑ spikes when new subtopic introduced (brain, spinal cord, PNS...)
```

### Narrative: Moon Landing → 1 branch, immediate exhaustion

**Seed:** *"On July 20, 1969, Neil Armstrong descended the ladder of the lunar module Eagle and stepped onto the surface of the Moon."*

**Output (32 tokens, 1 branch):**

> He uttered his famous words that would echo, "That's one small step for man, one giant leap for mankind."

**Supersaturation:** `0.41` — below threshold after one branch. Stories have a single narrative thread; there's nothing to branch into.

### Comparative: Mitosis vs Meiosis → 20 branches, *rising* supersaturation

**Seed:** *"Mitosis and meiosis are both forms of cell division, but they serve different biological purposes."*

The most striking example: supersaturation **rises** across 20 branches as the model discovers more comparison points.

```
0.52 → 0.53 → 0.58 → 0.65 → 0.74 → 0.79 → 0.81 → 0.85 → ...
                                               ↑ self-sustaining growth
```

This "rising supersaturation" phenomenon is unique to comparative texts — the parallel structure creates a self-reinforcing engine where each new contrast point generates pressure for the next.

## Supersaturation Taxonomy

Across 200 seeds (40 per discourse type), shorter 32-token branches, entropy-based supersaturation:

| Discourse Type | Avg Branches | Multi-branch % | Avg Tokens |
|----------------|-------------|-----------------|------------|
| **Comparative** | **3.3** | **37.5%** | 96 |
| Expository | 1.9 | 30.0% | 47 |
| Causal | 1.6 | 17.5% | 38 |
| Narrative | 1.2 | 10.0% | 32 |
| Argumentative | 1.1 | 10.0% | 27 |

Significant pairwise differences (paired t-tests):
- Argumentative vs Comparative: *d* = −0.62, *p* = 0.007
- Expository vs Argumentative: *d* = +0.60, *p* = 0.010
- Narrative vs Comparative: *d* = −0.59, *p* = 0.012

## Algorithm

```
function dendritic_generate(seed_text):
    text = seed_text
    for i in 1..MAX_BRANCHES:
        # Grow a new branch via masked diffusion
        branch = denoise(text + [MASK] * BRANCH_LEN)
        text = text + clean(branch)

        # Measure supersaturation (entropy-based)
        sat = measure_entropy(text, position=end)
        if sat < THRESHOLD:
            break  # crystal has reached equilibrium

    return text
```

The key insight: **the model determines its own output length.** No fixed token budget, no stop token heuristics — the crystal grows until the thermodynamic signal says it's done.

## Running

Requires a GPU with ≥15GB VRAM (tested on NVIDIA T4).

```bash
pip install torch transformers bitsandbytes accelerate scipy

# Run the taxonomy experiment
python code/sat_taxonomy_v3.py

# Run a single dendritic generation
python code/dendritic_v6b_base.py --prompt "Your seed text here"
```

## Connection to L-Systems

| L-System Concept | Dendritic Diffusion Analog |
|-----------------|---------------------------|
| Bracketed L-system (push/pop) | Branch start/stop via supersaturation threshold |
| Context-sensitive grammar | Supersaturation depends on generated context |
| Stochastic productions | Temperature sampling during denoising |
| Growth functions | Supersaturation decay curves |
| Axial trees (Horton-Strahler) | Discourse hierarchy (topic → subtopic → detail) |
| Parametric L-systems | Feature-guided generation (future work) |

## Citation

Paper in preparation. If you use this code, please cite:

```bibtex
@article{balogh2026dendritic,
  title={The Algorithmic Beauty of Text: L-System Grammars for Dendritic Language Generation},
  author={Balogh, Peter},
  year={2026},
  note={In preparation}
}
```

## License

MIT
