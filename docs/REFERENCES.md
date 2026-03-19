# Dendritic Diffusion — Key References

## Foundational Physics & Morphology

### Philip Ball — *Nature's Patterns: Branches* (Oxford UP, 2009)
Third volume of the *Tapestry in Three Parts* trilogy. Covers the physics of branching patterns across nature: snowflake dendritic growth, crack propagation, river networks, biological vasculature, and network topology.

**Key concepts for our work:**
- **Marginal stability in dendritic growth** (Ch. 1): Crystal tip growth speed balances against the tendency to split. There's a unique "marginally stable" pattern where growth just outpaces splitting instability. *This is the regime our supersaturation threshold should target.*
- **Laplacian growth / DLA** (Ch. 4): Diffusion-limited aggregation produces branching river networks. The growth front moves where the gradient is steepest — analogous to denoising where the highest-confidence tokens crystallize first.
- **Bifurcation cascades** (Epilogue, Fig 7.8): As a system is driven harder, patterns undergo period-doubling bifurcations → eventually chaos. Our supersaturation traces may show analogous transitions: steady growth → oscillation → repetition collapse.
- **Murray's law** (Ch. 5): Optimal branching in biological vasculature — branch diameters follow cube-root scaling. Could inform token allocation across branches.
- **"Palette of principles, not a Law of Pattern"** (Epilogue): Small changes in initial conditions produce dramatically different patterns from the same few processes. Exactly what we see with different seed texts.
- **Noise and pattern selection** (Epilogue): Noise doesn't affect all patterns equally — it may favor some over others. Connects to our temperature sampling favoring certain crystallization paths.

**Local copy:** `~/clawd/media/inbound/` (Nature's Patterns: Branches)

---

### Prusinkiewicz & Lindenmayer — *The Algorithmic Beauty of Plants* (Springer, 1990)
The definitive reference on L-systems (Lindenmayer systems) for modeling plant development. Provides the formal grammar framework for generating branching structures through parallel rewriting.

**Key concepts for our work:**
- **L-systems as parallel rewriting** (§1.1-1.2): At each step, ALL symbols are rewritten simultaneously. This IS masked diffusion — all masked tokens updated in parallel. L-systems are the formal grammar that describes what diffusion models do.
- **Bracketed OL-systems** (§1.6.3): `[` pushes turtle state onto stack, `]` pops it. Creates branching by save/restore. Direct implementation path for dendritic generation: `[` = checkpoint crystal state, grow side branch, `]` = return to main trunk.
- **Axial trees** (§1.6.1): Formalism for ordered branching. At each node, one outgoing segment is "straight" (main continuation), others are "lateral" (side branches). Maps to: main narrative continuation vs. elaborative digressions.
- **Context-sensitive L-systems** (§1.8): Productions that depend on neighboring symbols. Maps to: supersaturation measurement, where what grows next depends on the existing crystal context.
- **Stochastic L-systems** (§1.7): Probabilistic production rules — different random choices produce different but structurally similar organisms. Maps to: temperature sampling producing different crystallization paths from the same seed.
- **Parametric L-systems** (§1.10): Productions carry numerical parameters (growth rate, branch angle, etc.). Maps to: supersaturation, confidence, semantic/structural energy as parameters governing branch decisions.
- **Growth functions** (§1.9): Control growth rate and timing across the organism. Maps to: supersaturation decay curves controlling when branches stop growing.
- **DOL-systems and Fibonacci** (§1.2): The simplest L-system `a→ab, b→a` produces Fibonacci growth. Could text generation exhibit similar growth patterns?
- **Edge rewriting vs. node rewriting** (§1.4): Two strategies for where new structure appears. Edge rewriting = expanding existing content. Node rewriting = growing from branch points. Our dendritic approach is node rewriting (new content grows from the crystal tip).
- **Horton-Strahler ordering** (§1.6.1): Method for classifying branch hierarchy in stream networks and trees. Could classify discourse structure depth.

**Local copy:** `~/clawd/media/inbound/` (ABOP)
**Also available:** http://algorithmicbotany.org/papers/#abop

---

## Diffusion Language Models

### LLaDA (GSAI-ML, 2025)
- Base: `GSAI-ML/LLaDA-8B-Base` — our primary model (no SFT, no EOS bias)
- Instruct: `GSAI-ML/LLaDA-8B-Instruct` — v5e model (EOS attractor problem)
- Paper: arXiv:2502.09992

### Dream (Dream-org, 2025)
- Base: `Dream-org/Dream-v0-Base-7B` — tested in v6 comparison, severe degeneration
- AR-initialized weights, adaptive remasking strategy
- Paper: arXiv:2503.07573

### Related Papers (tracked in diffusion-sae-research.md)
- **CoDD** (2603.00045): Breaks factorization barrier with coupled output distributions
- **DOS** (2603.15340): Dependency-oriented sampling via attention matrices
- **MetaState** (2603.01331): Persistent working memory across denoising steps
- **DyLLM** (2603.08026): Most tokens stable across steps — saliency-based skipping
- **Skip to the Good Part** (2603.07475): dLLMs have MORE hierarchical abstractions than AR

---

## Crystal Growth & Supersaturation

- Mullins-Sekerka instability — why flat growth fronts become dendritic
- Constitutional supercooling — how supersaturation gradients drive branching
- Ivantsov parabola — steady-state tip shape of dendrites
- See: Langer, "Instabilities and pattern formation in crystal growth" (Rev. Mod. Phys. 1980)
