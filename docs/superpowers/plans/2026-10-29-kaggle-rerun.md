# Kaggle AML rerun implementation plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. Use regression tests for changes to experimental correctness.

**Goal:** Rebuild AML features from raw IBM HI-Small data and rerun a reproducible augmentation comparison on private Kaggle GPU kernels.

**Architecture:** Kaggle builds features and exports training positives plus baseline results. LLM generation remains local and its caches are uploaded for the final comparison; configuration files and credentials are never bundled. Each run uses a separate output directory and manifest.

**Tech Stack:** Python, pandas, NumPy, XGBoost, SMOTENC, CTGAN, Kaggle CLI.

## Requirements and decisions

- User authorized a fresh rerun and Kaggle CLI execution. API cap is USD 2 at user-provided prices: USD 0.1 / million input tokens and USD 0.5 / million output tokens.
- Proposed comparison: E0, random duplication ER, SMOTENC E1, CTGAN E2, LLM-basic and LLM-AML; seeds 0, 1, 2; ratio 0.5 to fit the cap.
- Rebuild from raw; do not replace historical local caches or results.
- Fix timestamp units, self-transfer history, receiver amounts, and timestamp ties at split boundaries. Record the feature version and exact split statistics.
- Treat same-timestamp transactions within a split in stable raw-file order, an explicit observational-order assumption. No timestamp occurs in two splits.
- Require exact synthetic sample counts and matching feature metadata. Never silently substitute a smaller cache.
- Report AP, threshold metrics and precision/recall at a fixed top-1000 alert budget. Select thresholds on validation only.
- Do not publish, commit secrets, or send credentials to Kaggle code.

## Tasks

### Task 1: Correct features and temporal split
- [x] Write failing regression tests for datetime units, self-transfer snapshots, received amounts, timestamp-boundary splits and stale cache detection.
- [x] Fix src/data.py and add explicit versioned metadata; use a new cache filename.
- [x] Run the feature regression suite.

### Task 2: Reproducible experiments
- [x] Test strict LLM cache counts, random duplication, count constraints and fixed alert-budget metrics.
- [x] Add ER, scaled SMOTENC numeric distances, lazy optional dependencies, common repair of generated feature invariants, strict LLM cache loading.
- [x] Add isolated result directories, predictions, training-positive export, environment/source manifest and resume validation.
- [x] Run regression tests and a small end-to-end CPU smoke run.

### Task 3: Kaggle bundle and launch
- [x] Test portable POSIX bundle paths, code allowlisting, privacy, configurable owner, and syntax of generated kernel.
- [x] Generate a private GPU kernel for baseline reruns and training-positive export; pin the environment used in its manifest.
- [x] Authenticate, push, monitor and download results when user credentials are available.

### Task 4: Budgeted LLM comparison
- [x] Implement local resumable generation for basic and AML prompts using Kaggle-exported training positives.
- [x] Enforce shared USD 2 budget including conservative reservations, disable SDK retries and checkpoint every paid call.
- [x] Validate caches and upload only samples plus non-secret metadata for the final Kaggle comparison.
- [x] Retrieve results and produce a Vietnamese analysis with limitations.

## Progress

- Read-only diagnostics complete. Kaggle CLI 2.2.5 installed in .venv. Credentials requested asynchronously.
- Work remains in the user-shared checkout; no worktree or branch mutation is needed for rerun artifacts.
- Tasks 3 launch and 4 depend on Kaggle authentication. No paid API requests have been made.
- Task 1: complete; timestamp, self history, received amount, shared currency codebook, split groups and stale/raw-input cache regressions pass.
- Task 2: complete for E0/ER/E1/E2, with exported positives, predictions and isolated provenance. Important review findings on opposite-direction counts and shared self histories fixed RED -> GREEN before GPU push.
- Task 3: version 1 pushed to private snnasnguyen/aml-rerun-20261029, T4 x2 confirmed in live logs; baseline E0/ER/E1 executed, CTGAN running as of last log read.
- Updated actual split: train 3,046,861 / 2,297 positives; val 1,015,602 / 1,082; test 1,015,882 / 1,798. Python round(0.5 * 2297) gives 1,148 generated rows for every method.
- Task 4: budget ledger, locked paired prompts, strict validator and cache loader implemented; no paid API calls yet. Awaiting exported positives and code review.
- Ruling: Kaggle v1 uses the pre-raw-hash-validation reader, but always rebuilds from raw; later reader adds identity validation without changing feature values. Preserve v1 submission as provenance.
- Baselines COMPLETE; downloaded to results/rerun_kaggle_baselines/results/rerun_20261029. All 12 NPZ files verified to share labels and reproduce CSV AP. Summary and plots: results/rerun_analysis_baselines.
- Paired LLM stage review fixed CRLF hash transfer, paid-response replay, atomic CSV/meta checkpoints, and binding LLM cache hashes to experiment manifests. 29 local regression tests pass.
- Actual provider rejected max_tokens (HTTP 400). Changed to max_completion_tokens as explicitly required by API; retained both old conservative reservations. No samples were returned by those two calls.
- Live LLM processes: exec session 12619 (pilot seed 0, both variants, target100, workers2); exec session 99707 (full seeds1/2, target1148, workers4). Both use data/llm/rerun_20261029/budget.sqlite and the same immutable generator source.
- Next: when pilot session12619 finishes, run the SAME generate_llm_rerun.py with --seeds 0 --workers 2 and default ratio .5, reusing pilot caches/checkpoints; do not change generator source while its processes run.
- Then wait for all six caches to reach1148 rows, validate feature fingerprints, build a SECOND private Kaggle kernel with --llm-dir data/llm/rerun_20261029 and --args "--exps E3-basic E3-AML --seeds 0 1 2 --ratio 0.5 --rebuild-features --output-dir results/rerun_llm_20261029". Snapshot v1 is preserved under results/rerun_kaggle_baselines/submission_v1.
- Pilot session12619 COMPLETE (100 rows per variant). Seed0 expansion is now running in session56268 (workers2); seeds1/2 remain in session99707 (workers4).
- Unattended finisher running in session89109: kaggle/finish_rerun.py --owner snnasnguyen. It waits for all6888 rows, validates all caches and baseline-core AST equivalence, submits snnasnguyen/aml-llm-rerun-20261029 on private T4, downloads results, runs analyze_results.py, and writes results/rerun_analysis/report_vi.md. Live log: results/rerun_analysis/pipeline.log. State/lock are in the same directory.
- Latest observed progress at watcher start:979/6888 generated rows. Budget cap unchanged at USD2. Do not modify generate_llm_rerun.py while generation processes run: checkpoints bind its hash.
- This task remains IN PROGRESS until results/rerun_analysis/pipeline_state.json contains complete=true. Baseline results are already complete and independently verified.
- COMPLETE: all6888 LLM rows generated; all18 detector runs finished. Private LLM notebook: snnasnguyen/aml-llm-rerun-20261029. Controller exited successfully and pipeline_state.json has complete=true.
- Final verification independently recomputed AP from all six LLM NPZ files, checked shared test labels against baseline, exact1148 synthetic rows per run, identical train-positive export hashes, compatible manifests and18 unique experiment/seed rows.
- Final mean AP: E0 .670024; ER .662307; E1 .672593; E2 .647953; E3-basic .608909; E3-AML .596862. LLM prompts underperform baseline in this configuration. AML-minus-basic is positive on2/3 seeds but mean negative, so prompt effect is unstable.
- Estimated usage cost at user-provided prices: USD1.0531798 settled; USD1.0658262 including the two conservative reservations from rejected max_tokens calls. No further paid requests needed.
- Final deliverables: results/rerun_analysis/report_vi.md, table.md, all_results.csv, paired_ap_differences.csv, average_precision.png/pdf, api_budget.json, baseline_core_equivalence.json. Historical outputs remain separate.
