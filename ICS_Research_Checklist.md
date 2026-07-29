# Comprehensive Research Checklist
## Hierarchical Local–Global Register-State Anomaly Detection for Multi-PLC ICS

Tick every box → strong conference-ready paper in hand.

---

## PHASE 0 — Setup & Access

- [*] **Submit SWaT (A1/A2, Dec 2015) access request to iTrust** — takes 1–3 weeks, do this first, before anything else. Confirm academic email + short proposal are ready.
- [*] **Identify and download a secondary register-only dataset** (WADI or HAI, most recent version with the most attack scenarios) — needed for generalization claims (Addition #4).
- [*] **Set up the 3-PLC testbed for logging** — confirm you can read/log register values at consistent sample rate (match SWaT's 1Hz if possible, for comparability).
- [*] **Set up attack injection capability on the testbed** — confirm you can issue Modbus write commands (FC5/6/15/16) bypassing the HMI, and can script repeatable attack runs.
- [ ] **Provision the Raspberry Pi deployment target** — install runtime dependencies (Python, relevant ML libraries), confirm it can receive register streams from the testbed in real time or near-real-time.
- [ ] **Assign the 5-person team to workstreams** (local models, global models, fusion, testbed/data, evaluation/writing) with a shared task tracker.

---

## PHASE 1 — Data Preparation

- [ ] **Preprocess SWaT physical data**: clean, label, normalize (fit scaler on train-normal only), tag→PLC mapping, train/val/test split.
- [ ] **Preprocess secondary dataset (WADI/HAI)** the same way, using an identical pipeline structure to SWaT so comparisons are apples-to-apples.
- [ ] **Preprocess your own testbed data**: same cleaning/labeling/normalization pipeline, once attack runs are collected.
- [ ] **Define and document your 3-PLC subset mapping** (which SWaT stages/PLCs correspond to your 3 local nodes, and why they're physically connected) — needed to justify your global model's graph structure.
- [ ] **Design and document attack scenarios in 3 categories**: local-only-detectable, global-only-detectable, both-detectable — with multiple variants (magnitude/duration/timing) per category for statistically meaningful counts.
- [ ] **Map every attack scenario (testbed + dataset attacks) to MITRE ATT&CK for ICS technique IDs** *(Addition #5)* — grounds your evaluation in a recognized threat taxonomy; strengthens realism claims and gives reviewers a familiar reference frame.
- [ ] **Obtain the SWaT P&ID (Piping and Instrumentation Diagram) and process description from iTrust** — authoritative ground truth for which tags/stages are physically connected; typically included alongside dataset access.
- [ ] **Document the P&ID / process flow diagram for your own 3-PLC testbed** — you designed it, so formalize it now; needed for methodology section regardless of other uses.
- [ ] **Empirically validate/discover inter-stage connections statistically** — run lagged cross-correlation (Pearson/Spearman), Granger causality tests, and/or transfer entropy or mutual information between tag pairs across PLCs, accounting for process delay (e.g., water transit time between stages). Compare against the documented P&ID: do they agree? Does the data reveal connections the documentation misses? This is a citable sub-result, not just a preprocessing step.
- [ ] **Define the graph edge set for the global model** using the P&ID as primary ground truth, refined/cross-checked by the statistical analysis above.

---

## PHASE 2 — Local Layer Models

- [ ] **Implement Markov-chain + dwell-time bounds + sequence likelihood model** (your primary/novel local model) — per-PLC transition matrices, empirical dwell-time distributions, thresholding.
- [ ] **Implement PASAD or EPASAD** (subspace/SSA-based SOTA local model) — window length + subspace dimension tuned per key sensor tag.
- [ ] **Implement register-invariant rule model** — hand-derive a small set of physical/logical invariants per PLC (classical baseline, interpretable by construction).
- [ ] **Implement LSTM-Autoencoder** (deep-learning baseline) — reconstruction-error-based anomaly scoring.
- [ ] **Implement Isolation Forest** (naive floor baseline) on per-PLC feature windows.
- [ ] **Train/tune all 5 local models identically** across all 6 stages, using the same train/val split and threshold-setting procedure for every model.
- [ ] **Run all 5 local models against every dataset** (SWaT, secondary dataset, own testbed).

---

## PHASE 3 — Global Layer Models

- [ ] **Implement GAT (Graph Attention Network)** over your 3-PLC process graph (your primary/novel global model) — define edges based on physical process connectivity.
- [ ] **Implement centrality-aware GCN (CGAAD-inspired)** — compute centrality measures (degree/betweenness) via networkx, incorporate as node weighting/features in a GCN.
- [ ] **Implement VAR (Vector Autoregression)** — interpretable classical baseline, explicit cross-PLC coefficients.
- [ ] **Implement LSTM-Autoencoder**.
- [ ] **Implement Isolation Forest** on the joint (all-PLC) feature vector (naive floor baseline).
- [ ] **Train/tune all global models identically**, same split/thresholding procedure as local layer.
- [ ] **Run all global models against every dataset.**

---

## PHASE 4 — Architectural Controls & Ablations (Critical — this is what proves your thesis)

- [ ] **Implement a flat/non-hierarchical baseline model** *(Addition #6)* — a single monolithic model (e.g., one LSTM or one GAT) trained on all 3 PLCs' combined data with no local/global split. This is your control condition: if your hierarchical architecture doesn't clearly outperform this, your core architectural claim is unproven. Non-negotiable inclusion.
- [ ] **Run local-only vs. global-only vs. fused evaluation** — the central ablation table of the paper.
- [ ] **Run per-attack-category breakdown** (SSSP/SSMP/MSSP/MSMP or your own local/global/both categories) for every model and for local-only/global-only/fused — shows exactly which layer catches what.
- [ ] **Document specific cases where global explains away a local false positive** (and vice versa) — qualitative evidence supporting the fusion narrative, good for a case-study figure.

---

## PHASE 5 — Fusion Layer (Your Core Novel Contribution — invest real time here)

- [ ] **Formalize the fusion algorithm explicitly** *(Addition #2)* — decide and document: rule-based/time-window correlation, weighted-scoring, cascade/trigger model, or learned fusion (e.g., logistic regression/Dempster-Shafer over the two layers' scores). Give it a name.
- [ ] **Implement at least 2 fusion strategies for comparison** (e.g., simple rule-based AND a weighted/learned version) — an ablation *within* the fusion layer itself, not just a single fixed design.
- [ ] **Evaluate fusion strategies against local-only and global-only baselines** on precision/recall/F1/AUPRC and false-alarm rate.
- [ ] **Report detection latency improvement (or cost) from fusion** — does correlating across layers slow detection down, and is the accuracy gain worth it?

---

## PHASE 6 — Adversarial / Evasion Testing

- [ ] **Design 5–10 adversarially-crafted attack variants** *(Addition #3)* specifically constructed to try to stay under both the local and global anomaly thresholds simultaneously (e.g., slow-drift attacks tuned to known threshold values, attacks that mimic legitimate cross-PLC correlation patterns).
- [ ] **Run these adversarial attacks against local-only, global-only, and fused configurations** — report how much harder they are to evade under fusion vs. either layer alone. This is a strong, reviewer-convincing test of your core "why both layers" argument.
- [ ] **Discuss remaining evasion risk honestly** — no detection system is unbeatable; a candid limitations discussion here builds credibility rather than undermining it.

---

## PHASE 7 — Deployment & Systems Evaluation (Raspberry Pi)

- [ ] **Deploy the best-performing model(s) per layer to the Raspberry Pi** in the 3-PLC testbed.
- [ ] **Measure compute cost**: CPU/memory usage, inference time per sample (ms), model size/parameter count, per model.
- [ ] **Measure end-to-end detection latency** in the real deployment (not just offline inference time) — from attack injection to alert.
- [ ] **Measure throughput limits** — how many concurrent PLC streams can the Pi handle before degrading, relevant to real-world scaling claims.
- [ ] **Compare deployed (Pi) performance against offline/development-machine performance** — confirm no accuracy degradation from quantization, resource constraints, etc., or report the tradeoff if there is one.
- [ ] **Implement correlation-filtered forwarding**: local nodes send only the tags with a known/discovered inter-stage connection (per Phase 1's connection analysis) to the global node, instead of all tags.
- [ ] **Measure bandwidth/communication cost**: full-tag forwarding vs. correlation-filtered forwarding — bytes transmitted per cycle, network utilization.
- [ ] **Confirm detection performance does not degrade** under correlation-filtered forwarding vs. full-tag forwarding — report side-by-side F1/AUPRC comparison.
- [ ] **Discuss attack-surface reduction from filtered forwarding** — fewer inter-node data channels between local and global layers means fewer interception/spoofing points on the monitoring infrastructure itself, distinct from the ICS process attack surface.

---

## PHASE 8 — Interpretability Evaluation

- [ ] **Pick and commit to one concrete interpretability method** *(Addition #7)* — e.g., SHAP value analysis for tree/deep models, attention-weight visualization for GAT, direct rule inspection for invariants/Markov chain, or a small structured review with domain-knowledgeable evaluators rating explanation clarity. Don't leave this vague.
- [ ] **Apply the chosen method consistently across all models being compared** — same interpretability metric/method for every model in the comparison table.
- [ ] **Produce at least one interpretability case study figure** — e.g., "here's what the model flagged and why, in human-readable terms," for one real detected attack.

---

## PHASE 9 — Model Comparison Framework (Reframe from "leaderboard" to "decision framework")

- [ ] **Build a full comparison table**: every model (local + global) × metrics (F1, AUPRC, FPR, latency, compute cost, interpretability score) × dataset.
- [ ] **Run statistical significance testing** *(Addition #8)* across model comparisons — multiple seeds/runs for stochastic models (LSTM-AE, GAT, GCN), paired significance tests where claiming one model beats another.
- [ ] **Synthesize results into a decision framework, not just a winner** *(Addition #6 reframing)* — e.g., "use Markov+dwell-time if latency-constrained and interpretability is required; use GAT if maximum accuracy is needed and compute is available; use invariants when zero training data is available." This is a stronger, more citable takeaway than a single "best model" claim.

---

## PHASE 10 — Generalization Validation

- [ ] **Run the full pipeline (best local model + best global model + fusion) on the secondary dataset (WADI/HAI)** *(Addition #4)* — confirm results hold on a different physical process, not just SWaT.
- [ ] **Report cross-dataset generalization results explicitly** — e.g., train on SWaT, test on own testbed (or vice versa) if feasible, to test overfitting to one process's dynamics.
- [ ] **If generalization is weak, report and discuss it honestly** rather than omitting — this is a legitimate, citable limitation, not a fatal flaw, if handled transparently.

---

## PHASE 10.5 — Additional Hierarchical-Architecture Justifications (Discussion-section material)

- [ ] **Write up bandwidth/attack-surface reduction as a formal justification** (measured in Phase 7) — the monitoring architecture itself has a smaller, more defensible attack surface than a flat/all-data-forwarded design.
- [ ] **Write up the scalability argument** — a flat/monolithic global model's input dimensionality grows with plant size and eventually becomes computationally infeasible; hierarchical local pre-filtering keeps the global layer's input manageable regardless of plant size. Frame explicitly as "designed to scale beyond the 3-PLC evaluation testbed" since this isn't directly testable at your scale.
- [ ] **Write up the incremental deployment argument** — local nodes can be deployed/protect a PLC immediately; new PLCs can be added via a new local node + graph update without retraining every existing local model, unlike a flat model requiring full retraining.
- [ ] **Write up the fault isolation / diagnosability argument** — hierarchical alerts localize the problem (local-only vs. cross-PLC inconsistency) rather than a single flat anomaly score with no locality information; tie this directly to the Phase 8 interpretability case study to show hierarchical models give more actionable alerts than the flat baseline.
- [ ] **Write up the heterogeneous retraining-cadence argument** — local models are cheap to retrain frequently (e.g., after maintenance changes one PLC's normal behavior) without needing to retrain the more expensive global model every time.

---

## PHASE 11 — Writing & Packaging

- [ ] **Write Introduction** — motivate the local/global gap, state the flat-baseline comparison and adversarial testing as evidence, not just claims.
- [ ] **Write Related Work** — cover invariant-based detection, HAMIDS-style hierarchical network IDS, GNN/GAT-based global models, PASAD/EPASAD, DeepFM/CGAAD-adjacent work, digital-twin approaches — position your gap clearly (hierarchical register-only architecture with formalized fusion, validated with adversarial testing and edge deployment).
- [ ] **Write Architecture section** — local/global/fusion design, justification for why local is still needed given global (cite the resilience/single-point-of-failure literature discussed earlier).
- [ ] **Write Experimental Setup** — datasets, testbed description, attack scenario design + MITRE ATT&CK mapping, evaluation protocol, Raspberry Pi deployment specs.
- [ ] **Write Results** — all comparison tables (Phase 9), ablations (Phase 4), fusion evaluation (Phase 5), adversarial results (Phase 6), deployment/compute metrics (Phase 7), interpretability case study (Phase 8), generalization results (Phase 10).
- [ ] **Write Discussion/Limitations** — register-only scope (no network layer, stated as future work), evasion residual risk, generalization caveats if any.
- [ ] **Write Conclusion & Future Work** — network-layer fusion as natural extension, any other open threads.
- [ ] **Internal review pass** — at least one full team read-through before submission, checking every claim in the abstract/intro is actually backed by a results table.
- [ ] **Check target venue's formatting/page limits and submission requirements** — confirm early, not the week of deadline.
- [ ] **Prepare supplementary material** (code repo, dataset access notes, extended tables) if the venue allows/expects it — increasingly common and expected for reproducibility credibility.

---
