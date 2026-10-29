# Causal feature study for the AML master's course project

User approved the feature-engineering direction and asked to conduct experiments autonomously. Reuse the shared checkout and private Kaggle workflow; preserve all earlier artifacts. No LLM API calls or new paid services.

## Question and approaches
Does transaction history, recent activity, and local account relationships improve transaction-level detection under a fixed alert budget? The accepted approach is interpretable feature extraction plus the existing XGBoost, not a new GNN or another synthetic-data campaign. A GNN adds scope and confounds representation/model changes; augmentation has already been evaluated separately.

## Protocol
- IBM HI-Small, original stable ordering, existing timestamp-disjoint chronological 60/20/20 boundaries.
- All new history queries use timestamps strictly less than the query timestamp. Same-minute transactions never observe one another; current transaction fields remain available.
- Five predeclared configurations: F0 raw (11 columns); F1 raw + strict historical controls; F2 F1 + activity/amount windows; F3 F1 + local relationships; F4 all. Seeds 0,1,2. No augmentation or hyperparameter search; reuse XGB_PARAMS. All planned models are evaluated, with no test-driven feature selection.
- History controls reproduce the concepts of the existing 17 historical columns with strict timestamp ties. Historical mean-log-amount controls pool currencies as before; new monetary aggregates and deviations are computed within account/currency, never summing different currencies. A self-transfer contributes one outgoing and one incoming event after the whole timestamp batch is queried.
- Windows are [t-3600,t) and [t-86400,t). Numeric counts, currency-specific sums, and log-amount deviations belong to activity. Unique directional partners, prior reverse-pair interactions, relationship age/recency, pair-window counts, and partner shares belong to relationships. New-account missing histories remain NaN for means/recency, zero for counts/sums.
- Labels are never inputs to feature construction. Use account IDs as keys only, never direct model inputs. Do not add external KYC/FX/geo information absent from the source.
- Save validation and test predictions, validation-chosen F1 thresholds, feature lists/importances, per-run timing, source/raw hashes and split statistics. AP is primary; fixed top-1000 precision/recall and F1 secondary.
- Report all configurations and paired seed deltas. Three seeds measure training variability, not independent test populations. Previously inspected HI-Small test is exploratory reused evaluation, not a fresh confirmatory holdout. Do not claim real-bank performance or causation from feature importance.

## Deliverables
New feature module, regression tests with hand-computed and brute-force oracles, experiment runner, allowlisted private Kaggle bundle, downloaded results, Vietnamese report and standalone figures. Keep earlier pipelines unchanged except documentation/ignore entries.
