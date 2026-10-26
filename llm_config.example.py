"""LLM (OpenAI / ChatGPT API) configuration for E3.

WARNING: this file holds the API key -> do not commit it to git / share it.
kaggle/build_kernel.py only bundles src/*.py and run_experiments.py, so this file is NOT pushed to Kaggle.
"""
import os

# ======================= FILL IN YOUR API KEY HERE =======================
OPENAI_API_KEY = ""  # e.g. "sk-proj-..."; or leave empty and run: export OPENAI_API_KEY=...
# =========================================================================

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None = default OpenAI endpoint
TEMPERATURE = 1.0       # set to None if the model does not support temperature (e.g. reasoning models)
ROWS_PER_CALL = 25      # number of transactions generated per API call
EXAMPLES_PER_CALL = 20  # number of real transactions shown as few-shot examples per call
MAX_WORKERS = 8         # number of parallel API calls


def get_api_key():
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("No API key set: fill in OPENAI_API_KEY in llm_config.py "
                         "or export OPENAI_API_KEY=...")
    return key
