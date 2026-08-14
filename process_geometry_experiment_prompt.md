# Research Project: What Geometry Do Predictive Process Monitoring Models Learn?

You are acting as a senior machine-learning researcher, process-mining researcher, research software engineer, and experimental scientist.

Your task is to **design, implement, execute, analyze, and document a complete reproducible research experiment** investigating the latent geometry learned by state-of-the-art predictive process monitoring (PPM) models.

The intended output is a research paper targeting **ICPM 2027 — International Conference on Process Mining**, in the ML4PM workshop.

Do not treat the hypotheses below as facts to be demonstrated. They are hypotheses to be **tested and potentially falsified**. Negative results are scientifically meaningful.

---

# 1. Research motivation

Modern predictive process monitoring models are generally compared using task-specific predictive metrics:

- next-event accuracy / F1;
- suffix similarity;
- remaining-time MAE;
- outcome AUC/F1;
- calibration.

However, neural PPM models also learn latent representations of process prefixes.

For a trace

\[
\sigma = (e_1,e_2,\ldots,e_T),
\]

a model implicitly or explicitly maps every prefix

\[
h_t=(e_1,\ldots,e_t)
\]

to a latent representation

\[
z_t=f(h_t)\in\mathbb R^d.
\]

A complete case therefore induces a trajectory through latent space:

\[
z_1\rightarrow z_2\rightarrow\cdots\rightarrow z_T.
\]

Existing PPM evaluations largely ignore the geometry of these trajectories.

The central question of this project is:

> **How do different predictive objectives and architectures shape the latent geometry of event-driven processes, and does this geometry provide information about process behaviour, prediction reliability, and model explainability beyond conventional predictive metrics?**

This study is intended as the first step toward a broader research programme on **geometric representation learning and the geometry of possible futures in event-driven systems**.

Do NOT introduce a new representation-learning architecture in the main experiment unless required for a controlled baseline. The primary contribution should first establish what geometry existing models already learn.

---

# 2. Main research questions

Investigate the following RQs.

## RQ1 — Geometry

**What geometric structure emerges in the latent representations learned by state-of-the-art PPM models?**

Compare:

- next-event prediction;
- full-suffix prediction;
- outcome prediction where meaningful;
- simple/self-supervised representation baselines if useful.

Determine whether predictive objective affects:

- trajectory straightness;
- smoothness;
- curvature;
- displacement;
- latent velocity;
- trajectory efficiency;
- clustering;
- branch structure;
- terminal-state organization;
- intrinsic dimensionality;
- local neighborhood structure.

## RQ2 — Prediction vs geometry

**Is better predictive performance associated with particular geometric properties?**

Do NOT assume that straighter representations are better.

Test relationships between geometric metrics and:

- prediction correctness;
- confidence;
- calibration;
- suffix accuracy;
- remaining-time error;
- outcome prediction;
- trace predictability.

Determine whether two models with similar predictive performance can exhibit substantially different latent geometries.

## RQ3 — Reliability

**Does latent geometry contain information about prediction errors beyond conventional confidence measures?**

Test whether geometric characteristics can distinguish:

- correct vs incorrect predictions;
- high-confidence correct vs high-confidence incorrect predictions;
- easy vs difficult prefixes;
- in-distribution vs unusual cases where appropriate.

In particular test whether:

\[
P(\text{error}\mid\text{confidence},\text{geometry})
\]

can be estimated more accurately than:

\[
P(\text{error}\mid\text{confidence}).
\]

This is one of the project's highest-priority experiments.

## RQ4 — Process dynamics and explainability

**Can changes in latent geometry identify meaningful events and turning points in process execution?**

Investigate whether individual events correspond to:

- large latent displacement;
- large direction changes;
- movement toward/away from outcome regions;
- transitions between latent regions;
- branching or convergence.

Evaluate whether such geometric explanations correspond to actual process behavior rather than merely attractive visualizations.

## RQ5 — Future organization

**Do models trained on different predictive objectives organize states according to their possible futures?**

Test whether prefixes having similar future continuations are geometrically close even when their histories differ.

Conversely, test whether prefixes with similar histories but substantially different futures become separated.

This motivates the concept of **future equivalence**:

\[
h_i\sim_F h_j
\iff
P(\tau\mid h_i)\approx P(\tau\mid h_j).
\]

Determine empirically whether next-event and suffix models differ in their approximation of future-equivalent representations.

---

# 3. Core hypotheses

Test rather than assume:

### H1
Full-suffix supervision produces more globally organized latent trajectories than next-event supervision.

### H2
Next-event supervision produces representations optimized for local discrimination but potentially less coherent over complete trajectories.

### H3
Outcome supervision, where evaluated, produces strong terminal/outcome separation but may discard intermediate process dynamics.

### H4
Trajectory geometry provides predictive information about model errors beyond model confidence/entropy.

### H5
Large latent direction changes correspond disproportionately to meaningful process events, deviations, rework, or future bifurcations.

### H6
Models differ in how early their latent spaces separate cases leading toward qualitatively different futures.

### H7
Geometric organization reflects future similarity, not merely activity/prefix similarity, in the strongest representations.

Explicitly report rejected hypotheses.

---

# 4. Models

Perform a literature and GitHub review before implementation.

Prioritize models satisfying:

1. peer-reviewed or strong research provenance;
2. publicly available implementation;
3. reproducible training;
4. applicability to standard event logs;
5. accessible internal prefix representations.

At minimum include:

### A. Strong next-event baseline

Use a Transformer-based next-event PPM model with public implementation where possible.

Consider ProcessTransformer or a more recent reproducible SOTA model discovered during the literature review.

### B. Strong full-suffix baseline

Prioritize **SuTraN / SuffixTransformerNetwork**, subject to verifying that it remains an appropriate competitive baseline.

Use its official/public implementation where feasible.

### C. Controlled Transformer baseline

Implement a common Transformer encoder trained separately with:

- next-event objective;
- outcome objective if labels can be defined meaningfully.

This controlled comparison is essential because differences between published models otherwise confound **architecture** and **training objective**.

### D. Optional additional models

Add models only when scientifically useful, e.g.:

- LSTM PPM baseline;
- another recent SOTA suffix predictor;
- self-supervised sequence encoder;
- representation-learning baseline.

Do not create an unnecessarily large benchmark.

Aim for approximately 3–6 meaningful model configurations.

---

# 5. Critical experimental principle: separate architecture from objective

There are two comparisons.

## Experiment family A — ecological/SOTA comparison

Compare published models as they actually exist.

Question:

> What geometries do successful real PPM architectures learn?

## Experiment family B — controlled comparison

Keep encoder architecture, embedding dimension, data preprocessing and training budget as constant as reasonably possible while changing the learning objective.

For example:

\[
\text{Transformer}_{next}
\]

versus

\[
\text{Transformer}_{suffix}
\]

versus

\[
\text{Transformer}_{outcome}.
\]

This allows conclusions about **predictive objectives rather than architectural differences**.

Keep these analyses clearly separated in the paper.

---

# 6. Datasets

Use established public process-mining event logs.

Select approximately 4–6 datasets covering different structural properties.

Candidates include suitable BPI Challenge logs and other established public event logs.

The final selection should include variation in:

- number of cases;
- number of activities;
- trace length;
- trace variability;
- degree of branching;
- loops/rework;
- outcome diversity;
- process structuredness.

For each dataset report:

- cases;
- events;
- activities;
- median/mean trace length;
- trace variants;
- variant entropy;
- relevant attributes;
- temporal span;
- outcome definition if used.

Avoid cherry-picking datasets based on whether they produce visually attractive geometry.

Use identical train/validation/test splits across models.

Prevent temporal/data leakage.

Where feasible use multiple random seeds.

---

# 7. Representation extraction

For every trained model and every test case, extract a representation for every prefix:

\[
z_t=f(e_{1:t}).
\]

Clearly document which internal representation is used:

- CLS/state token;
- last-token representation;
- pooled encoder representation;
- decoder/context representation;
- etc.

Representations should preferably be extracted **before the final prediction head**.

For every trace store:

\[
Z_\sigma=(z_1,z_2,\ldots,z_T).
\]

Also store:

- true events;
- predicted events/suffix;
- prediction probabilities;
- entropy;
- outcome;
- remaining time;
- prefix length;
- case metadata;
- prediction correctness.

Do not rely on 2D UMAP/t-SNE coordinates for quantitative geometry.

All primary geometric metrics must be calculated in the original latent space.

Dimensionality reduction is for visualization only.

---

# 8. Geometry metrics

Implement a modular geometry-analysis package.

## 8.1 Trajectory straightness

For trace \(\sigma\):

\[
S(\sigma)=
\frac{\|z_T-z_1\|_2}
{\sum_{t=1}^{T-1}\|z_{t+1}-z_t\|_2}.
\]

Interpret cautiously.

High straightness alone is NOT evidence of representation quality.

## 8.2 Path length

\[
L(\sigma)=
\sum_{t=1}^{T-1}
\|z_{t+1}-z_t\|_2.
\]

Normalize where necessary for trace length and embedding scale.

## 8.3 Latent velocity / event displacement

\[
v_t=\|z_t-z_{t-1}\|.
\]

Interpret as magnitude of representation update caused by event \(e_t\).

Investigate whether large displacement corresponds to meaningful process events.

## 8.4 Local direction change / curvature proxy

For consecutive displacement vectors:

\[
\Delta_t=z_t-z_{t-1}
\]

compute:

\[
\theta_t=
\arccos
\frac{\Delta_t^\top\Delta_{t+1}}
{\|\Delta_t\|\|\Delta_{t+1}\|}.
\]

Handle numerical degeneracies carefully.

Investigate:

- mean curvature;
- maximum curvature;
- curvature distribution;
- event-level curvature.

## 8.5 Smoothness / acceleration

Define:

\[
a_t=
\|(z_{t+1}-z_t)-(z_t-z_{t-1})\|.
\]

Analyze whether smoothness correlates with prediction reliability.

## 8.6 Progress toward terminal regions

Construct terminal-state prototypes or distributions where meaningful.

For outcome \(y\):

\[
c_y=
\mathbb E[z_T\mid Y=y].
\]

Measure:

\[
d_y(t)=d(z_t,c_y).
\]

Do not assume Euclidean prototypes are always valid. Also evaluate neighborhood/distribution-based alternatives.

Measure whether distance to the eventual outcome region decreases as the case progresses.

## 8.7 Branch separation

For groups leading to different outcomes/future variants, measure separation as a function of normalized prefix progress.

Determine when future groups become statistically/geometrically distinguishable.

Define an empirical **geometric predictive horizon** or bifurcation time only if supported by robust statistical analysis.

## 8.8 Future-equivalence neighborhood quality

Define future similarity independently of latent embeddings using observable suffixes.

Candidate distances:

- normalized edit distance;
- Damerau-Levenshtein;
- activity-set similarity;
- n-gram similarity;
- outcome similarity;
- remaining-time similarity;
- learned suffix distance only as a secondary measure.

For prefix pairs \(i,j\), compare:

\[
d_Z(z_i,z_j)
\]

against:

\[
d_F(\tau_i,\tau_j).
\]

Measure:

- rank correlation;
- nearest-neighbor retrieval quality;
- precision@k for future-similar states;
- trustworthiness/continuity where appropriate.

Control for similarity of the observed histories.

A particularly important experiment is:

> Among prefixes with dissimilar histories, does latent proximity retrieve cases with similar futures?

And conversely:

> Among prefixes with similar histories, does latent distance increase when their futures diverge?

---

# 9. Guard against representation collapse and trivial geometry

Straightness can be artificially high if:

\[
z_1\approx z_2\approx\cdots\approx z_T.
\]

Therefore always pair trajectory metrics with representation-quality diagnostics.

Include:

- embedding variance;
- covariance spectrum;
- effective rank;
- pairwise distance distribution;
- intrinsic dimensionality;
- terminal-state separability;
- neighborhood preservation.

A representation must not be labelled geometrically superior merely because trajectories are straight.

A desirable geometry should preserve **meaningful variation and branching**.

---

# 10. Prediction metrics

Retain conventional PPM evaluation.

For next-event prediction:

- accuracy;
- macro-F1;
- top-k accuracy if justified;
- NLL/cross-entropy;
- calibration/ECE;
- Brier score where applicable.

For suffix prediction:

- normalized Damerau-Levenshtein similarity or metric standard in the selected baseline literature;
- suffix length error;
- remaining-time MAE if model predicts it.

For outcome prediction:

- AUROC;
- AUPRC;
- macro-F1;
- calibration.

Use the metrics standard to the original benchmark implementations whenever possible to maintain comparability.

---

# 11. Critical reliability experiment

This is a centerpiece of the paper.

For each prediction construct features including:

Baseline reliability information:

- maximum predicted probability;
- predictive entropy;
- margin between top predictions;
- prefix length.

Geometric information:

- straightness-so-far;
- recent curvature;
- maximum curvature;
- recent latent displacement;
- cumulative path length;
- smoothness;
- distance to training manifold / nearest-neighbor distance;
- local density;
- distance to predicted outcome/future region where appropriate.

Fit simple, interpretable error-prediction models on validation data:

### Model A

\[
P(error\mid confidence)
\]

### Model B

\[
P(error\mid confidence,prefix\ length)
\]

### Model C

\[
P(error\mid confidence,prefix\ length,geometry).
\]

Evaluate on held-out test data using:

- AUROC;
- AUPRC;
- Brier score;
- calibration;
- likelihood improvement where appropriate.

Use paired statistical tests and confidence intervals.

The key question:

> Does geometry provide incremental information about prediction failure beyond conventional model uncertainty?

Also specifically analyze **high-confidence errors**.

---

# 12. Explainability / event-importance experiment

For each event transition:

\[
z_{t-1}\rightarrow z_t,
\]

calculate:

- displacement;
- direction;
- curvature;
- movement relative to terminal/future regions.

Rank events by geometric impact.

Then test whether high-impact events correspond to:

- activity transitions associated with different outcomes;
- rework;
- deviations;
- rare events;
- known process milestones;
- future suffix changes.

Avoid relying solely on qualitative examples.

Quantify associations.

Where appropriate compare geometric event importance against:

- attention scores;
- gradient-based attribution;
- SHAP or another established explanation baseline.

Investigate whether geometry answers a distinct question:

> **Which observed event changed the model's internal conception of where the process is heading?**

Do not claim causal effects.

Use terms such as "representation shift" or "model-internal turning point" unless causal identification is explicitly performed.

---

# 13. Turning points

Explore two concepts separately.

## Trajectory turning point

An event producing a large change in latent direction:

\[
\theta_t\gg0.
\]

## Future turning point

An event after which the distribution/structure of observed future outcomes changes substantially.

Do not assume they coincide.

Test whether geometric turning points predict:

- outcome divergence;
- prediction changes;
- uncertainty changes;
- suffix-family changes.

This distinction may be important for subsequent Future-Space research.

---

# 14. Statistical methodology

Do not draw conclusions from UMAP plots.

Use:

- bootstrap confidence intervals;
- paired tests across cases where applicable;
- effect sizes;
- correction for multiple comparisons where necessary;
- mixed-effects/regression models when datasets introduce hierarchical structure;
- seed-level variation.

Report distributions rather than only means.

Treat dataset as an important source of variation.

Test whether findings replicate across multiple event logs.

---

# 15. Controls and ablations

At minimum investigate:

### Random/untrained encoder

Determines whether geometric patterns arise simply from sequence structure.

### Representation layer

Compare selected internal layers for at least one architecture.

### Embedding normalization

Check robustness to:

- raw representations;
- L2 normalization;
- whitening where justified.

### Distance metric

Compare at least:

- Euclidean;
- cosine/angular distance.

Consider Mahalanobis or learned metrics only if scientifically justified.

### Prefix length

Ensure geometry findings are not trivial consequences of longer prefixes.

### Trace length

Ensure straightness/path metrics are not artifacts of sequence length.

### Activity frequency

Ensure event displacement isn't merely a proxy for rarity.

### Multiple seeds

Verify that geometric findings are stable under model retraining.

---

# 16. Visualization

Create publication-quality visualizations including, where useful:

1. UMAP/PCA trajectory plots for representative cases;
2. trajectories colored by normalized process progress;
3. trajectories colored by outcome;
4. geometric metric distributions by model;
5. geometry vs prediction-error plots;
6. branch-separation curves over prefix progress;
7. reliability comparison with/without geometry;
8. event-level curvature/displacement annotated on traces;
9. future-similarity vs latent-distance plots;
10. effective-rank / representation-spectrum diagnostics.

Use PCA alongside nonlinear visualizations because UMAP/t-SNE can create misleading apparent structure.

Never use visualizations as primary evidence for a quantitative claim.

---

# 17. Expected possible findings

Do NOT force these conclusions.

Possible outcomes include:

### Finding A

Suffix-prediction models produce smoother/straighter and more future-organized representations than next-event predictors.

### Finding B

Next-event models are locally discriminative but globally fragmented.

### Finding C

Outcome models strongly cluster terminal outcomes but lose detailed trajectory structure.

### Finding D

Geometry predicts errors beyond softmax confidence.

This would support geometry as a reliability-monitoring signal.

### Finding E

High-curvature/displacement events correspond to meaningful process transitions.

This would support geometric process explainability.

### Finding F

Predictive performance and geometric organization are weakly correlated.

This would be particularly interesting because it suggests conventional metrics measure only one dimension of representation quality.

### Finding G

No robust relationship exists.

This is also valid and must be reported rather than hidden.

---

# 18. What NOT to claim

Do not claim:

- straightness is inherently desirable;
- Euclidean geometry is necessarily the correct geometry;
- latent displacement is causal event importance;
- UMAP clusters prove latent organization;
- a model is superior merely because its geometry is visually cleaner;
- future-space structure exists simply because a model predicts suffixes;
- geometric explanations correspond to human explanations without evaluation.

Use precise language.

Distinguish:

- observed geometry;
- predictive association;
- model interpretation;
- process interpretation;
- causal explanation.

---

# 19. Main scientific contribution

The intended contribution is NOT:

> "Model X has straighter embeddings."

The intended contribution is:

> **A systematic geometric characterization of representations learned by predictive process monitoring models, establishing whether latent trajectory geometry provides information about process dynamics and model reliability beyond conventional predictive metrics.**

A stronger contribution, if supported, is:

> **Different predictive objectives induce systematically different geometries, and geometric properties provide complementary signals for understanding process progression, future divergence, and prediction reliability.**

---

# 20. Longer-term research context

This paper should establish foundations for a subsequent research programme.

Future work may explicitly learn representations satisfying properties such as:

- future equivalence;
- coherent trajectory progression;
- branch preservation;
- semantic straightening;
- reachable-future organization;
- turning-point sensitivity.

Eventually the representation of a prefix

\[
z_t=f(h_t)
\]

could be associated with a structured future space

\[
\mathcal F(z_t)
\]

representing possible future trajectories.

This future programme should motivate the present work but should NOT dominate the current paper.

The current study asks first:

> **What geometry do existing predictive models already learn?**

---

# 21. Reproducibility requirements

Create a fully reproducible Git repository.

Suggested structure:

```text
process-geometry/
├── README.md
├── pyproject.toml
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   ├── representations/
│   ├── geometry/
│   ├── evaluation/
│   ├── reliability/
│   ├── explainability/
│   └── visualization/
├── experiments/
├── scripts/
├── notebooks/
├── results/
├── figures/
├── tests/
└── paper/
```

Use configuration-driven experiments.

Record:

- dataset version/hash;
- preprocessing;
- split;
- model configuration;
- random seed;
- software versions;
- checkpoint;
- git commit;
- hardware where relevant.

Cache extracted embeddings so geometric analyses can be rerun without retraining models.

Write unit tests for geometry metrics using synthetic trajectories for which expected values are analytically obvious.

---

# 22. Synthetic validation

Before analyzing real models, construct synthetic latent trajectories to validate every metric.

Include examples such as:

### Perfectly straight

\[
z_t=(t,0)
\]

Expected straightness = 1.

### Curved path

Points sampled from a circular arc.

### Backtracking

Trajectory moving toward a target and then reversing.

### Collapsed representation

\[
z_t=c.
\]

### Branching trajectories

Common path followed by two diverging branches.

### Reconverging trajectories

Branches separating and later merging.

Use these tests to demonstrate exactly what each metric measures and expose pathological cases.

---

# 23. Paper-oriented outputs

Maintain a continuously updated research report containing:

1. research questions;
2. hypotheses;
3. related work;
4. datasets;
5. models;
6. experimental protocol;
7. metric definitions;
8. statistical methodology;
9. results;
10. rejected hypotheses;
11. threats to validity;
12. discussion;
13. implications for geometric process representation learning.

Generate all paper tables and figures programmatically from experiment results.

Never manually copy numerical results into paper artifacts.

---

# 24. Required final analyses

At the end, explicitly answer:

### Q1
Which model/objective has the strongest predictive performance?

### Q2
Which has the most coherent trajectory geometry, according to each metric?

### Q3
Are prediction quality and geometric quality correlated?

### Q4
Does geometry improve prediction-error detection beyond confidence?

### Q5
Can geometric changes identify meaningful model-internal turning points?

### Q6
Which objective best organizes states according to similarity of their futures?

### Q7
Are the findings consistent across datasets?

### Q8
Which geometric metrics appear genuinely informative and which appear redundant or misleading?

### Q9
Is there enough evidence to justify explicitly optimizing geometry in a subsequent model?

### Q10
Do the results support a broader research direction around the geometry of possible futures?

Be willing to answer Q9 or Q10 with "not yet" if the evidence does not support them.

---

# 25. Execution strategy

Work incrementally.

## Phase 1 — Literature and repository audit

Search recent peer-reviewed literature and public repositories.

Produce a table containing:

- paper;
- year;
- venue;
- prediction task;
- architecture;
- datasets;
- GitHub repository;
- license;
- last update;
- reproducibility status;
- accessible representation layer;
- expected integration difficulty.

Use this evidence to finalize model selection.

Do not rely on model names in this prompt without verifying them.

## Phase 2 — Dataset pipeline

Implement common preprocessing and splits.

## Phase 3 — Baseline reproduction

Reproduce published predictive results within reasonable tolerance before performing geometric analysis.

If reproduction fails, document why.

## Phase 4 — Representation extraction

Create standardized prefix-level embedding datasets.

## Phase 5 — Geometry validation

Implement and test geometry metrics using synthetic trajectories.

## Phase 6 — Descriptive geometry study

Characterize all model/dataset combinations.

## Phase 7 — Reliability study

Test incremental predictive value of geometry for detecting errors.

## Phase 8 — Future-organization study

Evaluate future-equivalence and branch separation.

## Phase 9 — Explainability study

Analyze event displacement, curvature and turning points quantitatively and qualitatively.

## Phase 10 — Statistical synthesis

Aggregate results across datasets/models with uncertainty estimates.

## Phase 11 — Paper

Produce the ICPM/ML4PM workshop manuscript.

---

# 26. Decision rules

At every stage prefer:

- reproducibility over model count;
- strong baselines over many weak baselines;
- quantitative evidence over visual impressions;
- effect sizes over isolated p-values;
- controlled comparisons over architectural confounding;
- falsifiable hypotheses over confirmatory storytelling;
- simple metrics over unnecessary mathematical complexity;
- original latent dimensions over dimensionality-reduced coordinates.

If an experimental result contradicts the research hypothesis, preserve and analyze it.

---

# 27. First action

Do NOT immediately start implementing models.

First:

1. perform the literature/repository audit;
2. identify the strongest reproducible next-event and full-suffix models;
3. inspect how prefix representations can be extracted from each;
4. select datasets;
5. identify confounds;
6. refine the experimental matrix;
7. estimate computational requirements;
8. produce a concrete implementation plan;
9. identify which analyses are essential for the ICPM/ML4PM workshop paper and which should be deferred.

Then implement the smallest end-to-end experiment on **one dataset × two models**.

Validate representation extraction and geometric metrics.

Only after this pilot succeeds should you scale to the complete benchmark.

The final research project should answer not merely:

> "Which model predicts best?"

but:

> **"What does a good predictive process representation look like geometrically, and does knowing its geometry tell us something useful that predictive accuracy alone does not?"**
