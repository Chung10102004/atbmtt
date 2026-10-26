"""E3: generate synthetic laundering transactions with an LLM (OpenAI Chat Completions API).

The LLM generates in the *feature space* (same as SMOTE/CTGAN) so the comparison is fair.
Each call: show EXAMPLES_PER_CALL random real laundering transactions + column descriptions
+ statistics + AML typologies, and ask for ROWS_PER_CALL new transactions as JSON.

Contains no API key - the key lives in llm_config.py (root) and is passed in via `client`.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from data import CAT_COLS, ROOT

LLM_DIR = ROOT / "data/llm"

PAY_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro",
              "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", "Shekel", "Swiss Franc",
              "UK Pound", "US Dollar", "Yen", "Yuan"]

FEATURE_DOCS = {
    "log_amt_paid": "log1p(amount paid by sender)",
    "log_amt_recv": "log1p(amount received); equals log_amt_paid when currencies are the same",
    "hour": "hour of day 0-23",
    "weekday": "0=Monday ... 6=Sunday",
    "pay_format": "payment format code: " + ", ".join(f"{i}={v}" for i, v in enumerate(PAY_FORMATS)),
    "pay_currency": "payment currency code: " + ", ".join(f"{i}={v}" for i, v in enumerate(CURRENCIES)),
    "recv_currency": "receiving currency code (same coding as pay_currency)",
    "same_bank": "1 if sender and receiver are at the same bank",
    "self_txn": "1 if sender account == receiver account",
    "cross_currency": "1 if pay_currency != recv_currency",
    "round_amt": "1 if amount paid is a multiple of 100",
    "pair_prev_n": "number of earlier transactions from this sender to this receiver",
    "src_prev_n": "sender: number of earlier transactions (in + out) = src_prev_out + src_prev_in",
    "src_prev_out": "sender: number of earlier outgoing transactions",
    "src_prev_in": "sender: number of earlier incoming transactions",
    "src_prev_out_amt_mean": "sender: mean log1p amount of earlier outgoing txns (null if src_prev_out=0)",
    "src_prev_in_amt_mean": "sender: mean log1p amount of earlier incoming txns (null if src_prev_in=0)",
    "src_secs_since_prev": "sender: seconds since its previous transaction (null if src_prev_n=0)",
    "src_same_day_n": "sender: earlier transactions on the same calendar day",
    "src_same_day_dir_n": "sender: earlier OUTGOING transactions on the same day (<= src_same_day_n)",
    "dst_prev_n": "receiver: number of earlier transactions = dst_prev_out + dst_prev_in",
    "dst_prev_out": "receiver: number of earlier outgoing transactions",
    "dst_prev_in": "receiver: number of earlier incoming transactions",
    "dst_prev_out_amt_mean": "receiver: mean log1p amount of earlier outgoing txns (null if dst_prev_out=0)",
    "dst_prev_in_amt_mean": "receiver: mean log1p amount of earlier incoming txns (null if dst_prev_in=0)",
    "dst_secs_since_prev": "receiver: seconds since its previous transaction (null if dst_prev_n=0)",
    "dst_same_day_n": "receiver: earlier transactions on the same calendar day",
    "dst_same_day_dir_n": "receiver: earlier INCOMING transactions on the same day (<= dst_same_day_n)",
}

TYPOLOGIES = """Known laundering typologies in this (IBM AMLworld) simulation:
- fan-out: one account quickly sends many transfers to many accounts
- fan-in: many accounts send to one account in a short period
- cycle: money moves A -> B -> C -> ... -> A
- scatter-gather: one source fans out to intermediaries which all forward to one sink
- gather-scatter: many sources fan in to one hub which then fans out
- bipartite / stack: layers of intermediary accounts passing funds between groups
- random: a random walk of transfers through several accounts"""

SYSTEM_PROMPT = ("You are a financial-crime data simulation expert. You generate realistic synthetic "
                 "training data for an academic anti-money-laundering (AML) detection research project. "
                 "Always answer with valid JSON only.")


def column_stats(X_pos):
    lines = []
    for c in X_pos.columns:
        s = X_pos[c]
        if c in CAT_COLS:
            top = s.value_counts(normalize=True).head(5)
            lines.append(f"- {c}: " + ", ".join(f"{k}: {v:.0%}" for k, v in top.items()))
        else:
            q = s.quantile([0.05, 0.5, 0.95])
            lines.append(f"- {c}: p5={q.iloc[0]:.2f}, median={q.iloc[1]:.2f}, p95={q.iloc[2]:.2f}, "
                         f"null={s.isna().mean():.0%}")
    return "\n".join(lines)


def _rows_to_json(df):
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    df = df.astype(float).round(2)  # float64 first, otherwise float32 prints as 5.619999885
    df = df.astype(object).where(df.notna(), None)
    for c in int_cols:
        df[c] = df[c].map(int)
    return json.dumps(df.to_dict(orient="records"), separators=(",", ":"))


def build_prompt(X_pos, examples, n_rows, stats):
    docs = "\n".join(f"- {c}: {FEATURE_DOCS[c]}" for c in X_pos.columns)
    return f"""Each row below is ONE money-laundering transaction from a synthetic bank dataset, \
described by engineered features (history features only count transactions BEFORE this one).

Columns:
{docs}

Statistics of real laundering transactions:
{stats}

{TYPOLOGIES}

Real laundering examples:
{_rows_to_json(examples)}

Task: generate {n_rows} NEW, diverse laundering transactions that are plausible under the typologies \
above and consistent with the statistics. Do NOT copy the examples. Respect the constraints \
(prev_n = prev_out + prev_in, same_day_dir_n <= same_day_n, nulls as described, cross_currency \
consistent with currencies, log_amt_recv = log_amt_paid when currencies match). Add a field \
"typology" naming the typology each row follows.

Return JSON: {{"rows": [{{<all {len(X_pos.columns)} columns>, "typology": "..."}}, ...]}}"""


def parse_rows(text, columns):
    """Parse the LLM's JSON, keep only valid rows, and fix deterministic constraints."""
    try:
        rows = json.loads(text)["rows"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return pd.DataFrame(columns=[*columns, "typology"])
    df = pd.DataFrame([r for r in rows if isinstance(r, dict)])
    if df.empty or not set(columns) <= set(df.columns):
        return pd.DataFrame(columns=[*columns, "typology"])
    typ = df["typology"].astype(str) if "typology" in df else "unknown"
    df = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    df["typology"] = typ
    # Columns that must never be null
    required = [c for c in columns if "amt_mean" not in c and "secs_since" not in c]
    df = df.dropna(subset=required)
    df["cross_currency"] = (df["pay_currency"] != df["recv_currency"]).astype(int)
    for p in ("src", "dst"):
        df[f"{p}_prev_n"] = df[f"{p}_prev_out"] + df[f"{p}_prev_in"]
        df[f"{p}_same_day_dir_n"] = df[[f"{p}_same_day_dir_n", f"{p}_same_day_n"]].min(axis=1)
    return df.reset_index(drop=True)


def _call(client, model, prompt, temperature, seed, retries=5):
    kwargs = dict(model=model, response_format={"type": "json_object"}, seed=seed,
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}])
    if temperature is not None:
        kwargs["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content, resp.usage
        except Exception as e:  # rate limit / network: back off and retry
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  API error ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)


def generate(client, model, X_pos, n, seed, rows_per_call=25, examples_per_call=20,
             temperature=1.0, max_workers=8, log_dir=None):
    """Call the API in parallel until n valid rows are collected. Returns (DataFrame, stats)."""
    rng = np.random.default_rng(seed)
    stats = column_stats(X_pos)
    real_keys = set(map(tuple, X_pos.round(2).fillna(-1).to_numpy()))
    collected, usage = [], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "invalid_or_dup_rows": 0, "copied_rows": 0}

    def job(i):
        ex = X_pos.iloc[rng.choice(len(X_pos), examples_per_call, replace=False)]
        prompt = build_prompt(X_pos, ex, rows_per_call, stats)
        text, u = _call(client, model, prompt, temperature, seed * 100_000 + i)
        if log_dir is not None and i == 0:  # save one sample prompt/response for the report
            (log_dir / f"sample_prompt_seed{seed}.txt").write_text(prompt)
            (log_dir / f"sample_response_seed{seed}.json").write_text(text)
        return parse_rows(text, X_pos.columns), u

    n_have, call_id = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while n_have < n:
            n_calls = int(np.ceil((n - n_have) / rows_per_call * 1.15))  # +15% for invalid rows
            futures = [pool.submit(job, call_id + i) for i in range(n_calls)]
            call_id += n_calls
            for f in as_completed(futures):
                df, u = f.result()
                usage["calls"] += 1
                usage["prompt_tokens"] += u.prompt_tokens
                usage["completion_tokens"] += u.completion_tokens
                usage["invalid_or_dup_rows"] += rows_per_call - len(df)
                keys = df[X_pos.columns].round(2).fillna(-1).to_numpy()
                copied = np.array([tuple(k) in real_keys for k in keys], dtype=bool)
                usage["copied_rows"] += int(copied.sum())
                collected.append(df[~copied])  # drop rows that copy real data verbatim
            out = pd.concat(collected, ignore_index=True)
            out = out.drop_duplicates(subset=list(X_pos.columns)).reset_index(drop=True)
            n_have = len(out)
            print(f"  seed {seed}: {n_have}/{n} rows after {usage['calls']} calls")
    return out.iloc[:n].reset_index(drop=True), usage


def cache_path(seed):
    return LLM_DIR / f"llm_seed{seed}.csv"


def load_cached(seed):
    p = cache_path(seed)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run first: python generate_llm_data.py --seeds {seed}")
    return pd.read_csv(p)
