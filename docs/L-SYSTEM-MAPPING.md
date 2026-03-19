# L-System Mapping: From Plant Growth to Text Crystallization

## The Core Insight

L-systems and masked diffusion are the same thing viewed from different angles:

| L-systems | Masked Diffusion |
|-----------|-----------------|
| Alphabet of symbols | Vocabulary |
| String of symbols | Sequence of tokens |
| Parallel rewriting (all symbols at once) | Parallel denoising (all masks at once) |
| Productions (rewriting rules) | Learned logits (what each mask becomes) |
| Axiom (starting string) | Seed/prompt text |
| Derivation step | Denoising step |
| Generated organism | Generated text |

**The key difference**: L-systems apply explicit, hand-written rules. Diffusion models learn implicit rules from data. But the computational structure is identical: parallel rewriting of a string according to context-dependent rules.

**What L-systems add**: 60 years of theory about how branching, growth control, and hierarchical structure emerge from parallel rewriting. We can import this theory directly.

---

## Mapping 1: Bracketed L-Systems → Branching Generation

### L-system brackets
```
F[+F]F[-F]F
```
- `[` = push turtle state (save position + direction)
- `]` = pop (return to saved state)
- Content between brackets = side branch
- Content after `]` = main trunk continues

### Text generation equivalent
```python
# Current: linear branch-and-extend
crystal = seed + branch_0 + branch_1 + branch_2 + ...

# L-system inspired: tree-structured generation
crystal = seed + main_0 + [digression_0] + main_1 + [digression_1] + main_2
```

### Implementation: Stack-Based Dendritic Generation
```
1. Grow main branch (64 tokens, denoise)
2. Measure supersaturation at branch tip
3. If high semantic energy but topic divergence detected:
   PUSH crystal state → grow side branch → POP back to main trunk
4. If high energy and coherent continuation:
   Continue main branch (no push/pop)
5. If low energy:
   STOP (equilibrium)
```

**What "topic divergence" means**: The supersaturation probe shows high confidence but the predicted tokens are about a subtopic, not the main thread. E.g., writing about the immune system, the probe predicts confident tokens about "T cells" specifically — that's a side branch, not the main trunk about immune system overview.

### Experiment: v7_bracketed.py
- Grow main trunk with dendritic denoising
- At each branch point, measure whether high-sat predictions continue the main topic or diverge
- If diverge: push state, grow side branch (limited to N tokens), pop, continue main
- Compare: does tree-structured text read better than linear crystal growth?

---

## Mapping 2: Context-Sensitive L-Systems → Supersaturation as Context

### L-system context sensitivity
In a context-sensitive L-system (2L-system), productions depend on neighbors:
```
a < b > c → d    (b becomes d only if preceded by a and followed by c)
```

### Text generation equivalent
Our supersaturation measurement IS context sensitivity. What the model predicts for masked positions depends on the entire crystal context. The "production rule" for each position is conditioned on everything already crystallized.

But we can make this more explicit:

### Experiment: Context-Window Supersaturation
- Instead of measuring supersaturation only at the crystal TIP, measure it at multiple points along the crystal
- Find positions where supersaturation INCREASES mid-crystal (internal instability)
- These are natural branch points — where the text "wants" to elaborate
- Insert `[` (push) at high-sat interior points, grow elaboration, `]` (pop)

This is the Mullins-Sekerka instability from Ball's book: a flat growth front becomes dendritic when the supersaturation gradient varies along it. Interior high-sat points = where the flat text surface "bulges" into a branch.

---

## Mapping 3: Stochastic L-Systems → Temperature as Biological Variation

### L-system stochasticity
```
F → F[+F]F[-F]F    (probability 0.33)
F → F[+F]F         (probability 0.33)  
F → FF             (probability 0.34)
```
Same rules, different random choices → different but structurally similar organisms. A forest of trees from the same L-system, each unique but recognizably the same species.

### Text generation equivalent
Our temperature parameter (1.15) + top-k sampling is exactly stochastic L-system behavior. Same seed, different random choices during denoising → different but thematically coherent texts.

### Experiment: Forest Generation
- Run the same seed 10 times with temperature 1.15
- Measure structural similarity (branch count, branch lengths, supersaturation profiles)
- Do they form a recognizable "species"? (Same topic structure, different surface text)
- If yes: the L-system analogy is validated — our generation IS a stochastic L-system
- Visualize as a "forest" of discourse trees

---

## Mapping 4: Growth Functions → Supersaturation Decay

### L-system growth functions
L-systems can model differential growth rates: apical meristems grow faster than lateral ones. Growth functions control when each module activates.

### Text generation equivalent
Our supersaturation decay across branches IS a growth function:
```
Branch 0: sat=0.18 (high energy, fast growth)
Branch 1: sat=0.12 (moderate)
Branch 2: sat=0.09 (slowing)
Branch 3: sat=0.05 (equilibrium → stop)
```

The decay curve shape characterizes the "species" of text:
- **Exponential decay**: topic fully explored in a few branches (factual answer)
- **Plateau then drop**: sustained energy for N branches then exhaustion (narrative)
- **Oscillating**: energy rises and falls (argument with counterarguments)
- **Rising then falling**: topic builds to climax then resolves (story arc)

### Experiment: Supersaturation Taxonomy
- Run 50+ different seeds, record supersaturation traces
- Cluster the decay curves — do meaningful categories emerge?
- Do they correspond to discourse types? (Narrative, expository, argumentative, descriptive)
- Connect to Entry 93 (Schankian brainstorm): supersaturation fingerprints → operator types

---

## Mapping 5: Axial Trees → Discourse Hierarchy

### L-system axial ordering
Horton-Strahler ordering assigns hierarchy to branches:
- Order 0: main trunk (root to crown)
- Order 1: primary branches off trunk
- Order 2: secondary branches off primaries
- etc.

### Text generation equivalent
```
Order 0: Main argument/narrative thread
Order 1: Major supporting points / key scenes
Order 2: Examples, evidence, details for each point
Order 3: Specific data, quotes, micro-illustrations
```

This is exactly how well-structured expository text works. The L-system formalism gives us a way to GENERATE text with explicit hierarchical structure, not just hope it emerges.

### Experiment: Explicit Hierarchical Generation
```python
def generate_axial_tree(seed, max_order=2):
    # Order 0: main trunk
    trunk = grow_branch(seed, branch_tokens=128)  # longer for main thread
    
    # Find high-sat interior points (branch sites)
    branch_sites = find_interior_supersaturation_peaks(trunk)
    
    for site in branch_sites:
        # Order 1: primary branches
        context = trunk[:site]
        branch = grow_branch(context, branch_tokens=64)
        
        if max_order >= 2:
            # Order 2: sub-branches
            sub_sites = find_interior_supersaturation_peaks(branch)
            for sub in sub_sites:
                sub_branch = grow_branch(branch[:sub], branch_tokens=32)
    
    # Assemble into ordered tree, serialize depth-first
    return serialize_tree(trunk, branches)
```

---

## Mapping 6: Parametric L-Systems → Feature-Guided Generation

### L-system parameters
Parametric L-systems carry numerical values:
```
A(s) : s > threshold → F(s) [ +A(s*0.7) ] F(s) [ -A(s*0.7) ] F(s) A(s*0.9)
```
Each branch inherits parameters (scaled growth rate, branch angle) from its parent.

### Text generation equivalent
**Supersaturation IS the growth parameter.** But we could carry additional parameters:
- **Topic coherence score** (how close is this branch to the seed topic?)
- **Depth** (how many levels deep are we?)
- **Novelty** (n-gram overlap with existing crystal — our anti-repetition metric)

Each branch inherits and modifies these parameters:
```python
branch = grow(crystal, params={
    'sat': parent_sat * 0.9,       # energy decays
    'coherence': parent_coherence,  # topic must stay close
    'depth': parent_depth + 1,     # going deeper
    'novelty_threshold': 0.4,      # require novelty
})
```

---

## Priority Experiments (ordered by effort/impact)

1. **Forest Generation** (low effort, high insight) — Run same seed 10x, compare structures. Validates stochastic L-system analogy.

2. **Supersaturation Taxonomy** (medium effort, high insight) — Cluster decay curves from 50+ seeds. Maps discourse types to growth functions.

3. **Bracketed Generation v7** (medium effort, high impact) — Implement push/pop stack for side branches. First true tree-structured text generation.

4. **Interior Supersaturation Probing** (medium effort, high insight) — Find Mullins-Sekerka instability points inside the crystal. Natural branch sites.

5. **Explicit Hierarchical Generation** (high effort, high impact) — Full axial tree generation with Horton-Strahler ordering. The "real" architecture paper.

---

## Mapping 7: The Chomsky-Lindenmayer Split Recapitulated

### The historical parallel (March 17, 2026)

The AR-vs-diffusion debate in NLP recapitulates the Chomsky-vs-Lindenmayer split from the 1960s.

| Chomsky (1956) | Lindenmayer (1968) |
|----------------|-------------------|
| Sequential rewriting (one symbol per step) | Parallel rewriting (all symbols simultaneously) |
| Models competence (abstract structure) | Models growth (dynamic process) |
| Context-free → context-sensitive hierarchy | 0L → 1L → 2L hierarchy (same axis!) |
| Dependencies handled naturally (sequential) | Dependencies require context-sensitive extensions |
| Generates strings | Generates organisms |

| Autoregressive LMs | Diffusion LMs |
|--------------------|---------------|
| Sequential token generation (left to right) | Parallel denoising (all positions at once) |
| Dependencies via causal attention | Dependencies via bidirectional attention |
| Factorization is built in | **Factorization barrier** (CoDD, arXiv:2603.00045) |
| KV cache for efficiency | Cache reuse is hard (ES-dLLM, DyLLM) |

**The factorization barrier IS the Lindenmayer problem.** When you rewrite all symbols in parallel, you lose inter-symbol dependencies. Chomsky grammars handle this naturally because they're sequential. L-systems needed context-sensitive extensions (2L-systems) to recover dependencies. CoDD does exactly this — adds a probabilistic coupling layer, which is the NLP equivalent of upgrading from a 0L-system to a 2L-system.

**Why Lindenmayer rejected Chomsky:** Chomsky grammars can't model growth because they rewrite one symbol at a time while others wait. But cells don't wait — a plant grows everywhere simultaneously. Lindenmayer's parallel rewriting was biologically necessary.

**Why diffusion models exist for the same reason:** AR models generate one token at a time. But meaning doesn't unfold left-to-right — it crystallizes simultaneously across a document. An essay's introduction and conclusion are semantically coupled even before the middle is written. Diffusion's parallel generation captures this.

**Dendritic diffusion as the synthesis:** Our approach is neither pure Chomsky (fully sequential) nor pure Lindenmayer (fully parallel). It's a *depth-first L-system* — parallel within each branch, sequential across the branching structure. This mirrors how real plants grow: parallel cell division within each meristem, sequential activation of meristems along the growth axis.

**Use in paper:** This framing positions dendritic diffusion within a 60-year intellectual tradition. Reviewers who know formal language theory will immediately see the depth. Reviewers who don't will still get the intuition: "text grows like a plant, not like a ticker tape."

---

## The Paper This Becomes

**"The Algorithmic Beauty of Text: L-System Grammars for Dendritic Language Generation"**

Core thesis: Text generation via masked diffusion is formally equivalent to a stochastic, context-sensitive, parametric L-system. This equivalence imports 60 years of formal language theory and morphogenetic modeling into text generation. We demonstrate:
1. Supersaturation traces function as growth parameters
2. Anti-repetition detection corresponds to apical dominance
3. Tree-structured generation produces hierarchically organized text
4. The same "species" of L-system (same seed) produces structurally similar but surface-diverse texts
5. Discourse types correspond to distinct growth function families

Connects: Ball's branching physics, Prusinkiewicz's L-system formalism, our supersaturation finding, and the dendritic diffusion architecture.
