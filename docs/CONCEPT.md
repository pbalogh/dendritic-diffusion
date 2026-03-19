# Dendritic Diffusion: Emergent Tree Structure in Language Generation

**Title candidates:**
- "Dendritic Diffusion: Emergent Tree Structure in Language Generation"
- "Dendritic Diffusion: Growing Sentences from Supersaturation"
- "Growing Language: Depth-First Diffusion with Emergent Discourse Structure"

## 1. Core Thesis

Autoregressive language models generate text left-to-right with no mechanism for multi-scale planning or backtracking. Existing diffusion language models (LLaDA, Dream, MDLM) solve the directionality problem but generate all positions simultaneously — breadth-first — with no awareness of discourse structure.

We propose **Dendritic Diffusion**: depth-first generation where text grows like a crystal. Each branch crystallizes locally via masked diffusion, then its completion reveals the discourse structure it belongs to — retroactively creating parent nodes that constrain subsequent branches. Supersaturation (residual confidence at branch boundaries) drives continuation and decomposes into structural vs. semantic energy, providing a principled signal for when to grow, when to stop, and what kind of growth to produce.

The name is literal: *dendritic* means "tree-like" (Greek δενδρίτης). The generation process grows a tree — not by predicting the tree top-down, but by discovering it bottom-up during crystallization.

## 2. The Empirical Motivation

### 2.1 What Transformers Do Wrong

**Paper 1 — Anxiety of Influence (arXiv:2602.17526):**
Attention heads implement Bloom filters — fast, approximate membership tests with false positives. These false positives inject noise into the prediction signal.

**Paper 2 — Half the Nonlinearity Is Wasted (arXiv:2603.03459):**
~50% of MLP nonlinearity is spent on tokens that don't need it. Nonlinearity IS the disambiguation mechanism, but the budget is allocated uniformly rather than where it's needed.

**Paper 3 — Discrete Charm of the MLP (arXiv:2603.10985):**
MLP routing is binary: consensus holds (linear pass-through) or breaks (full nonlinear processing). This is a crude prediction-error detector that sometimes fails to fire.

**Darkness Visible (in preparation):**
L11's exception handler — 7 consensus neurons + exception core — is the model's attempt at backtracking. When it fails, wrong interpretations propagate as hallucinations.

**Knowledge Neurons experiment (2026-03-15):**
"Knowledge neurons" (Dai et al. 2022) are actually routing neurons — 36.5× enrichment, knockout INCREASES fact probability, 0/5 transplants succeed. Knowledge arrives via attention, not MLP storage. This means the MLP is doing routing/gating, not knowledge retrieval.

### 2.2 The Architectural Lesson

The transformer does predictive coding badly because it wasn't designed for it. The L11 exception handler is an emergency hack. What if generation itself were a refinement process — not commit-once-per-token, but iterative crystallization where structure emerges from the process?

## 3. The Architecture

### 3.1 Crystal Growth, Not Interpolation

**Why not Brownian Bridge?** This project originally proposed Brownian Bridge diffusion (interpolation between fixed endpoints). But the architecture that emerged is fundamentally different:

| | Brownian Bridge | Dendritic Diffusion |
|---|---|---|
| Starting point | Fixed endpoints | Prompt only (open-ended) |
| Structure | Prescribed by endpoints | Emerges during generation |
| Growth direction | Outside-in (endpoints → middle) | Depth-first (trunk → branches) |
| Termination | Interpolation completes | Supersaturation reaches equilibrium |
| Tree structure | Not applicable | Bottom-up discovery |

When endpoints ARE known (e.g., fill-in-the-middle), Brownian Bridge is a special case of Dendritic Diffusion. But open-ended generation — the common case — requires the growth process.

### 3.2 The Three Stages

**Stage 1 — Branch Crystallization:** Allocate masked tokens for a branch. Denoise via standard MDLM, conditioned on all previously crystallized content. The branch resolves locally — function words and structure first, content words after.

**Stage 2 — Supersaturation Measurement:** After crystallization, measure residual confidence on hypothetical continuation masks. Decompose into:
- **Structural supersaturation** (probability mass on function words, connectors, punctuation) → grammar demands completion
- **Semantic supersaturation** (probability mass on content words) → topic has more to say
- **Total supersaturation** → overall driving force for continued growth

**Stage 3 — Structure Discovery:** The supersaturation fingerprint reveals what discourse structure the branch belongs to:
- High semantic, concentrated on same-category items → **LIST/ENUMERATION**
- High structural, causal connectors active → **CAUSAL CHAIN**
- High semantic after general claim → **CLAIM-EVIDENCE**
- Structural energy on contrast words → **COMPARISON**
- Low supersaturation overall → **STANDALONE** (equilibrium, stop)

This discovery retroactively creates a parent node that constrains subsequent branches.

### 3.3 The Fundamental Innovation: Bottom-Up Tree Discovery

In every prior approach, structure is either absent (AR, standard diffusion) or predicted top-down before generation (plan-then-execute). Dendritic Diffusion discovers structure **during** generation.

The analogy is literal: a crystal's first face implies its crystal system. A hexagonal face tells you the whole crystal is hexagonal — you now know what other faces to expect. Similarly, a first sentence about "Paris as a capital" reveals an enumeration structure — you now know parallel sentences about other capitals should follow.

```
Step 1: Generate "Paris is the capital of France."
Step 2: Supersaturation fingerprint → high semantic, "London"/"Berlin" active
Step 3: Discovery: "I'm in a LIST of European capitals"
Step 4: Create parent [ENUMERATION: capitals, parallel structure]
Step 5: Next branch constrained by parent → parallel structure, different capital
```

See `EMERGENT_STRUCTURE.md` for full algorithm and fingerprint taxonomy.

### 3.4 Supersaturation as Dendritic Growth Factor

In biology, dendritic growth factors (neurotrophins like NGF, BDNF) signal dendrites to grow or retract. In our architecture, supersaturation plays the same role:

- **High supersaturation** → grow (nucleate new branch)
- **Low supersaturation** → stop (equilibrium reached)
- **Structural supersaturation** → extend current grammatical structure
- **Semantic supersaturation** → start new sentence on same topic
- **Neither** → the dendrite is complete

Empirically measured thresholds (LLaDA-8B, 2026-03-15):
- \> 0.4: Must continue (e.g., "Capital of France" = 0.586)
- 0.2-0.4: Context-dependent (e.g., "Photosynthesis" = 0.341)
- \< 0.2: Stop (e.g., "Sky is blue" = 0.136, "Water cycle" = 0.145)

## 4. Empirical Results (Phase 1, 2026-03-15)

### 4.1 Crystal Growth Visualization
Model: Qwen3-0.6B-diffusion-mdlm → LLaDA-8B-Instruct (int4, 5.8GB on T4)

**Finding:** "Paris" crystallizes before "The capital of France is ___" — content nucleates before scaffolding (at 0.6B). At 8B scale, function words crystallize first (avg step 14.5 vs content 35.4).

### 4.2 Depth-First vs Breadth-First (LLaDA-8B, 10 prompts)

| Metric | Breadth-First | Depth-First |
|---|---|---|
| Clean outputs | 8/10 | **9/10** |
| Quality issues | 2 | **1** |
| Avg sentences | 2.2 | 2.0 |
| Avg time | 12.9s | **4.7s (2.7× faster)** |

Qualitative: BF produces garbled multi-sentence text (repeated phrases, broken grammar). DF produces clean sentences because each branch has full attention + completed prior context.

### 4.3 Structural Crystallization Order (LLaDA-8B, 6 prompts)

Three findings from spaCy parse analysis of crystallization order:
1. **Function words crystallize first** (step 14.5 vs content 35.4) — structure before content
2. **Main clause before subordinate** (step 18.3 vs 24.1) — trunk before branches
3. **Root verbs resolve at step 0-2** — syntactic head crystallizes earliest

### 4.4 Supersaturation Decomposition (LLaDA-8B, 6 prompts)

Remaining energy at branch boundaries is overwhelmingly **semantic** (topic-level), not structural (grammar-level). The model finishes its grammar cleanly but topics have unresolved depth. The structural/semantic ratio shifts with complexity.

## 5. Connections to Our Other Work

- **Discrete Charm**: Binary MLP routing is a crude version of our supersaturation threshold. Consensus = low supersaturation (crystal at equilibrium). Exception = high supersaturation (needs processing).
- **Darkness Visible**: The 7 consensus neurons are crystal nucleation detectors. Their firing pattern IS the supersaturation measurement.
- **Knowledge neurons (today)**: Routing neurons don't store facts — they gate what flows through. This is exactly what dendritic branching does: route information along different branches.
- **Half the Nonlinearity**: Wasted nonlinearity = uniform compute for non-uniform supersaturation. Dendritic Diffusion allocates compute where supersaturation is highest.
- **Sense-stack polysemy**: Commitment to wrong sense = wrong crystallization direction. Depth-first prevents this because high-level structure constrains before ambiguous tokens resolve.
- **Schankian operators**: The supersaturation fingerprint after an event sentence should reveal the operator type. Different operators produce different discourse structures. The operator IS the crystal symmetry group.

## 6. What's Novel

1. **Depth-first diffusion**: All existing diffusion LLMs are breadth-first. We're the first to sequence branches.
2. **Emergent tree structure**: The discourse tree isn't predicted top-down — it's discovered bottom-up during generation. No prior work does this.
3. **Supersaturation as continuation signal**: Measurable, decomposable into structural vs. semantic components. Not a learned token or heuristic.
4. **Dendritic growth metaphor made literal**: The generation process IS dendritic growth, with supersaturation as growth factor and branch geometry implying the larger tree structure.
5. **Empirically grounded**: Derived from mechanistic findings about transformer failure modes, not architecture search.

## 7. Paper Outline

1. **Introduction**: The structure problem in language generation (AR = no structure, breadth-first diffusion = flat structure, we = emergent structure)
2. **Background**: Masked diffusion LLMs (LLaDA, Dream), crystal growth physics, dendritic processes
3. **Dendritic Diffusion**: The architecture — depth-first branching, supersaturation measurement, emergent tree discovery
4. **Experiments**: 
   - Crystal growth visualization (what resolves first?)
   - BF vs DF comparison (quality, speed, coherence)
   - Structural crystallization order (does syntax emerge?)
   - Supersaturation decomposition (structural vs semantic energy)
   - Emergent structure classification (LIST, CAUSAL, COMPARISON, etc.)
5. **Analysis**: Why depth-first works — branch conditioning prevents quality degradation
6. **Connection to neuroscience**: Dendritic computation, growth factors, bottom-up organization
7. **Discussion**: From fixed to emergent structure — implications for language generation
8. **Conclusion**: Text generation as crystal growth

## 8. Target Venues

- **NeurIPS 2026** (Sydney, Dec) — main track if Phase 2 results are strong
- **EMNLP 2026** (Budapest, Oct) — natural fit for the linguistic motivation
- **ICML 2026 workshops** — early Phase 1 results

## 9. Timeline

- **March 2026**: Architecture formalized, Phase 1 POC on EC2
  - ✅ 2026-03-15: Crystal growth experiments, BF vs DF comparison, structural analysis, supersaturation decomposition
  - ✅ Demonstrated: DF cleaner (9/10 vs 8/10), 2.7× faster, fewer issues
  - ✅ Discovered: function-first crystallization, main-clause-first, supersaturation as continuation signal
  - ✅ Named: "Dendritic Diffusion"
- **April 2026**: Emergent structure classifier, forced multi-branch experiments, SAE feature monitoring
- **May 2026**: Garden-path immunity test, larger-scale evaluation
- **June 2026**: Paper writing
- **July 2026**: Submit to NeurIPS or begin EMNLP preparation

## 10. Historical Connections

### Garden-Path Research → Dendritic Diffusion
- Garden-path immunity comes for free: depth-first resolution means main clause disambiguates before subordinate clause is generated
- Levy's surprisal maps to supersaturation: high surprisal = high supersaturation = more compute needed
- Hagoort's unification model is more naturally dendritic than autoregressive

### Predictive Coding Connection
The original Brownian Bridge framing connected to predictive coding (top-down predictions constrain bottom-up). Dendritic Diffusion preserves this but inverts the direction: bottom-up signals create top-down structures. This is closer to how development works in neuroscience — low-level patterns self-organize into higher-level representations.

### SAE Feature Hierarchy
SAE features at different granularities map to different levels of the dendritic tree:
- Macro-semantic features → trunk (topic, intent)
- Discourse features → major branches (claim, evidence, comparison)
- Syntactic features → minor branches (clause structure)
- Lexical features → leaves (word choice)

The noise schedule in future work could be SAE-aware: corrupt fine-grained features first, coarse last. Denoising then naturally proceeds from trunk to leaves — dendritic growth driven by feature hierarchy.
