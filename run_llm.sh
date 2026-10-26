#!/usr/bin/env bash
# E3: generate data with the ChatGPT API, then train + benchmark.
#   ./run_llm.sh            # generate locally, then run E0-E3 on Kaggle (GPU)  [default]
#   ./run_llm.sh local      # generate + run E3 locally (needs ~16GB RAM)
#   SEEDS="0" RATIO=0.5 ./run_llm.sh
# Fill in the API key in llm_config.py first.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
RATIO="${RATIO:-1.0}"
KERNEL=hoangducchung/aml-baseline-e0-e2

# 1) Call the API -> data/llm/llm_seed*.csv (seeds already generated are skipped)
$PY generate_llm_data.py --seeds $SEEDS --ratio "$RATIO"

if [[ "${1:-kaggle}" == "local" ]]; then
  # 2a) Run locally
  $PY run_experiments.py --exps E3 --seeds $SEEDS --ratio "$RATIO"
  cat results/summary.md
else
  # 2b) Run on Kaggle: rerun E0-E3 with the same settings for a clean comparison table
  $PY kaggle/build_kernel.py --args "--exps E0 E1 E2 E3 --seeds $SEEDS --ratio $RATIO"
  .venv/bin/kaggle kernels push -p kaggle/kernel
  echo "Waiting for the Kaggle kernel..."
  until .venv/bin/kaggle kernels status $KERNEL 2>/dev/null | grep -qiE "complete|error|cancel"; do sleep 60; done
  .venv/bin/kaggle kernels status $KERNEL
  rm -rf results/kaggle && mkdir -p results/kaggle
  .venv/bin/kaggle kernels output $KERNEL -p results/kaggle --force
  cat results/kaggle/results/summary.md
fi
