# AML causal feature study implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user accepted the feature study and authorized experiments; no additional design-approval loop is needed.

**Goal:** Complete a reproducible five-configuration, three-seed feature ablation on private Kaggle and explain the results for a master's security course.

**Architecture:** A separate strict-timestamp event scanner derives historical, recent-activity and relationship features. A separate runner reuses the original model and temporal split; a private allowlisted script executes and exports verifiable predictions. Local analysis checks predictions and produces the report.

**Tech Stack:** Python, NumPy/pandas, optional Numba acceleration, XGBoost, Kaggle CLI, matplotlib.

**Spec:** docs/superpowers/specs/2026-10-29-feature-study-design.md

## Global constraints
- Preserve all prior code/results and credentials; no paid API, no publication.
- Work in the existing user-shared checkout with separate files; .git is read-only in the harness, no commit or branch mutation.
- Strict timestamp histories; currency-specific new monetary aggregates; no labels/account IDs as predictive features.
- Fixed five configurations and seeds 0,1,2; same XGBoost parameters, split and validation-threshold policy.
- Explicitly disclose reused test and descriptive seed variability.

## Review focus
- Same timestamp and self-transfer history are queried before updating.
- Exact lookback boundaries, empty history, currency switching and receiver amounts.
- Relationships distinguish directional and reversed interactions, including self-transfers.
- Every prediction export matches metrics and identical test row identities.
- Kaggle bundle contains only allowlisted code, never .env or llm_config.py.

## Tasks
### Task 1: Feature extraction
- [x] Add failing tests in tests/test_feature_study.py for strict ties, self transfers, windows, currencies, permutations, prefix/label independence and a brute-force oracle.
- [x] Run to observe failure, implement src/feature_study.py, then pass tests.
### Task 2: Experiment and bundle
- [x] Add failing tests for group exclusions, private bundle and credential exclusion; implement run_feature_study.py and kaggle/build_feature_kernel.py.
- [x] Run a tiny local end-to-end smoke and full regression suite; build and submit the private GPU script.
### Task 3: Results and interpretation
- [x] Implement analyze_feature_study.py; verify prediction identities and independently recompute AP/threshold metrics.
- [x] Retrieve all 15 results, report paired differences and timing, export plots and Vietnamese report.
- [x] Obtain a focused independent review, address important issues and verify again.

## Ledger
- Planning: accepted approach reused from the preceding discussion; implement inline under the user's instruction to experiment.
- Ruling: strict timestamp controls make the main ablation internally fair; older v3 E0 remains historical context, not an interchangeable baseline.
- Task 1: Python and Numba 0.61.2 paths verified. Independent review identified subtractive monetary residues at realistic amount ranges; a RED regression was added, then compensated summation and exact empty-window resets implemented.
- Task 2: allowlisted bundle test and local five-configuration synthetic smoke passed. The local smoke uses only 15 trees and is excluded from research results.
- Private Kaggle v1 submitted: snnasnguyen/aml-feature-study-20261029. Source snapshot: results/feature_study_kaggle/submission_v1. Both T4 GPUs visible; feature build35s for5,078,345 rows; main training started. No submitted code changes after launch.
- Ruling: an exploratory per-day profile found a sharp prevalence/volume change from2022-09-11. Add explicitly post-hoc diagnostics on both fixed-test portions, without dropping rows, refitting, changing features or selecting configurations. Preserve all primary metrics.
- Analysis smoke verification completed; 43 local tests passed. Independent review found no Important/Critical issues in metric/provenance checks or report. Minor smoke-labeling issue fixed with a persistent banner. A RED/GREEN test also prevents confusing Recall@FPR1% with Recall@1000.
- COMPLETE: private Kaggle kernel status COMPLETE; downloaded50files (110,168,557bytes) under results/feature_study_kaggle/results/feature_study_20261029. Independent analysis verified15runs/30prediction files, fixed split labels, all reported metrics, validation-selected thresholds and source snapshot hashes. Raw hash matches local HI-Small exactly.
- Final AP means: F0 .170801, F1 .533886, F2 .661712, F3 .610407, F4 .684093. F4 vsF1 paired mean delta+.150208; F4 vsF2+.022381. F4 Precision@1000 .915 / Recall@1000 .508899. All3paired seeds improve F4 vsF1 and F4 vsF2; still descriptive, no population significance claim.
- Post-hoc fixed-test diagnostic: beforeSep11, F1 AP .333019 vsF4 .528256; tail contains1108rows/655positives (59.12%) and must be interpreted separately. F0 tailAP .973390 vsF4 .961001; gains are cohort-dependent.
- Final artifacts: results/feature_study_analysis/report_vi.md, table.md, verification.json, paired_differences.csv, temporal_diagnostics.csv, F4_feature_importance.csv, feature_ablation.png/pdf, daily_data_profile.csv. Method definitions: docs/feature-study-method.md. No paid API calls.
