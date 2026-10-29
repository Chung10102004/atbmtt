"""Generate two locked LLM prompt variants locally from Kaggle-exported train positives.

No full feature cache is loaded locally. Every API call reserves its worst-case
cost before sending, uses no automatic retries, and saves its response and usage.
"""
import argparse
import hashlib
import json
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import FEATURE_VERSION
from llm_budget import BudgetLedger
from llm_gen import FEATURE_DOCS, SYSTEM_PROMPT, TYPOLOGIES, _rows_to_json, column_stats
from llm_validate import training_fingerprint, valid_mask

MAX_OUTPUT_TOKENS = 9000


def atomic_text(path, contents):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents.encode("utf-8"))
    temporary.replace(path)


def prompt_for(x, seed, call_number, n_rows, variant):
    rng = np.random.default_rng(np.random.SeedSequence([seed, call_number]))
    examples = x.iloc[rng.choice(len(x), min(20, len(x)), replace=False)]
    typologies = "\n" + TYPOLOGIES if variant == "aml" else ""
    docs = "\n".join(f"- {c}: {FEATURE_DOCS[c]}" for c in x)
    return f"""Generate NEW positive-class money-laundering rows for an academic synthetic-bank benchmark.
Each row is an engineered feature vector, not an account graph. Histories count prior directional
events in stable transaction order, before the current transaction. Incoming means use amounts received.
Columns:
{docs}
Training-positive statistics:
{column_stats(x)}
{typologies}
Training examples (do not copy):
{_rows_to_json(examples)}

Generate exactly {n_rows} diverse rows consistent with the statistics and these rules:
- All counts are nonnegative integers; categories use only the documented integer codes.
- prev_n=prev_out+prev_in; same_day_n<=prev_n.
- src_same_day_dir_n counts outgoing today; dst_same_day_dir_n counts incoming today.
- Directional day counts cannot exceed lifetime directional counts, including the opposite direction.
- pair_prev_n<=min(src_prev_out,dst_prev_in).
- Each history amount mean is null exactly when its corresponding count is zero.
- secs_since_prev is null exactly when prev_n=0; otherwise a nonnegative multiple of 60.
- If no history, same_day_n=0. If gap=0, same_day_n>=1. If gap>=86400, same_day_n=0.
- There must be ONE possible minute in the given hour compatible with BOTH histories:
  a prior event after midnight means same_day_n>=1; before midnight means same_day_n=0.
- If self_txn=1, same_bank=1 and sender/receiver share identical prior counts, gaps and means;
  their same_day_dir_n values sum to the shared same_day_n.
- cross_currency equals (pay_currency!=recv_currency). For equal currencies, log amounts match.
Return JSON only: {{"rows":[{{<all 28 feature columns>,"typology":"optional descriptive label"}},...]}}.
"""


def request_batch(client, model, prompt, columns, seed, ledger, directory, logical_id="batch0"):
    # UTF-8 bytes bound ordinary byte-fallback tokenization; reserve extra room
    # for role framing. The completion limit includes generated reasoning tokens.
    input_bound = len(prompt.encode("utf-8")) + len(SYSTEM_PROMPT.encode("utf-8")) + 1024
    # The request protocol is part of journal identity. Earlier rejected
    # max_tokens calls retain their conservative reservations in the ledger.
    stem = directory / ("call_" + logical_id + "_completion_v1")
    request_path = stem.with_name(stem.name + "_request.json")
    response_path = stem.with_name(stem.name + "_response.json")
    identity = {"model": model, "seed": seed, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "system_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "max_completion_tokens": MAX_OUTPUT_TOKENS}
    if request_path.exists():
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if any(request.get(k) != value for k, value in identity.items()):
            raise ValueError("Existing paid-call journal has different inputs")
        if not response_path.exists():
            raise RuntimeError("Unresolved API call: reservation retained; inspect the journal before retrying")
        saved = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        call_id = ledger.reserve(input_bound, MAX_OUTPUT_TOKENS, str(directory / logical_id))
        atomic_text(request_path, json.dumps({**identity, "budget_call_id": call_id}, indent=2))
        atomic_text(stem.with_name(stem.name + "_prompt.txt"), prompt)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=1., seed=seed, max_completion_tokens=MAX_OUTPUT_TOKENS)
        saved = {"content": response.choices[0].message.content or "", "budget_call_id": call_id,
                 "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens,
                 "api_secs": round(time.perf_counter() - started, 3)}
        atomic_text(response_path, json.dumps(saved, ensure_ascii=False, indent=2))
    # Settlement is idempotent, so a journaled paid response is replayed after a crash.
    ledger.settle(saved["budget_call_id"], saved["prompt_tokens"], saved["completion_tokens"])
    text = saved["content"]
    try:
        raw = json.loads(text).get("rows", [])
        if not isinstance(raw, list):
            raw = []
        raw = [r for r in raw if isinstance(r, dict)]
        frame = pd.DataFrame(raw)
        if not set(columns).issubset(frame.columns):
            frame = pd.DataFrame(columns=[*columns, "typology"])
        else:
            typology = frame["typology"].astype(str) if "typology" in frame else "unspecified"
            frame = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
            frame["typology"] = typology
            frame = frame.loc[valid_mask(frame[list(columns)])].reset_index(drop=True)
    except (ValueError, TypeError, AttributeError):
        raw = []
        frame = pd.DataFrame(columns=[*columns, "typology"])
    return frame, {"calls": 1, "prompt_tokens": saved["prompt_tokens"], "completion_tokens": saved["completion_tokens"],
                   "returned_rows": len(raw), "rejected_rows": len(raw) - len(frame), "api_secs": saved.get("api_secs", 0.)}


def row_keys(frame):
    return [tuple(row) for row in frame.astype(float).round(2).fillna(-1).to_numpy()]


def generate_seed(client, model, x, n, seed, variant, output, ledger, call_limit):
    directory = output / variant / f"seed{seed}_calls"
    directory.mkdir(parents=True, exist_ok=True)
    csv = output / variant / f"llm_seed{seed}.csv"
    meta_path = csv.with_name(csv.stem + "_meta.json")
    checkpoint = csv.with_name(csv.stem + "_checkpoint.json")
    fingerprint = training_fingerprint(x)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {"feature_version": FEATURE_VERSION, "prompt_version": "paired-basic-aml-v1", "variant": variant,
                "seed": seed, "model": model, "train_fingerprint": fingerprint, "generator_sha256": source_hash}
    meta = {**identity, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "returned_rows": 0, "rejected_rows": 0, "copied_rows": 0, "duplicate_rows": 0, "rows": 0, "api_secs": 0.}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        meta = saved["metadata"]
        if any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot reuse LLM cache with different data, source, prompt or model")
        contents = saved["csv"]
        if hashlib.sha256(contents.encode("utf-8")).hexdigest() != meta["cache_sha256"]:
            raise ValueError("Checkpoint CSV hash is invalid")
        frame = pd.read_csv(io.StringIO(contents))
        if len(frame) != meta["rows"] or frame.columns.tolist() != [*x.columns, "typology"]:
            raise ValueError("Checkpoint schema or row count is invalid")
        # The single atomic checkpoint is authoritative if the export pair was interrupted.
        atomic_text(csv, contents)
        atomic_text(meta_path, json.dumps(meta, indent=2))
    elif csv.exists() or meta_path.exists():
        raise ValueError("Cache exists without an atomic checkpoint; inspect it before resuming")
    else:
        frame = pd.DataFrame(columns=[*x.columns, "typology"])
    real = set(row_keys(x))
    while len(frame) < n:
        if meta["calls"] >= call_limit:
            raise RuntimeError("Logical call limit reached; refusing further spending")
        call_number = meta["calls"]
        wanted = min(25, n - len(frame))
        prompt = prompt_for(x, seed, call_number, wanted, variant)
        batch, usage = request_batch(client, model, prompt, x.columns,
                                      seed * 100000 + call_number, ledger, directory,
                                      logical_id=f"{variant}_seed{seed}_batch{call_number:04d}")
        for k, value in usage.items():
            meta[k] += value
        copied = np.array([key in real for key in row_keys(batch[list(x.columns)])], dtype=bool)
        meta["copied_rows"] += int(copied.sum())
        batch = batch.loc[~copied]
        combined = batch.reset_index(drop=True) if frame.empty else pd.concat([frame, batch], ignore_index=True)
        duplicate = combined.duplicated(subset=list(x.columns))
        meta["duplicate_rows"] += int(duplicate.sum())
        frame = combined.loc[~duplicate].iloc[:n].reset_index(drop=True)
        meta["rows"], meta["requested_rows"] = len(frame), n
        meta["complete"] = len(frame) == n
        contents = frame.to_csv(index=False, lineterminator="\n")
        meta["cache_sha256"] = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        atomic_text(checkpoint, json.dumps({"metadata": meta, "csv": contents}, indent=2))
        atomic_text(csv, contents)
        atomic_text(meta_path, json.dumps(meta, indent=2))
        print(f"{variant} seed {seed}: {len(frame)}/{n}; budget committed ${ledger.total():.4f}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-positives", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/llm/rerun_20261029")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", choices=["basic", "aml"], default=["basic", "aml"])
    ap.add_argument("--ratio", type=float, default=.5)
    ap.add_argument("--rows-per-seed", type=int)
    ap.add_argument("--call-limit", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    metadata_path = args.train_positives.with_name(args.train_positives.stem + "_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.train_positives.read_bytes()).hexdigest()
    if metadata["feature_version"] != FEATURE_VERSION or metadata["sha256"] != digest:
        raise ValueError("Training-positive export has wrong version or hash")
    x = pd.read_csv(args.train_positives)
    if x.columns.tolist() != metadata["columns"] or len(x) != metadata["rows"]:
        raise ValueError("Training-positive schema/count differs from metadata")
    n = args.rows_per_seed if args.rows_per_seed is not None else round(args.ratio * len(x))
    if n <= 0:
        ap.error("Requested sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for variant in args.variants:
            (args.output_dir / f"dry_run_{variant}.txt").write_text(prompt_for(x, 0, 0, 25, variant), encoding="utf-8")
        print(f"Dry run: {n} rows per seed/variant; no API calls.")
        return
    import llm_config as cfg
    from openai import OpenAI
    ledger = BudgetLedger(args.output_dir / "budget.sqlite", 2., .1, .5)
    def task(seed, variant):
        with OpenAI(api_key=cfg.get_api_key(), base_url=cfg.OPENAI_BASE_URL, max_retries=0, timeout=120.) as client:
            return generate_seed(client, cfg.OPENAI_MODEL, x, n, seed, variant, args.output_dir, ledger, args.call_limit)

    jobs = list(dict.fromkeys((seed, variant) for seed in args.seeds for variant in args.variants))
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        futures = [pool.submit(task, seed, variant) for seed, variant in jobs]
        for future in as_completed(futures):
            future.result()
            atomic_text(args.output_dir / "budget_summary.json", json.dumps(ledger.summary(), indent=2))
    print(json.dumps(ledger.summary(), indent=2), flush=True)


if __name__ == "__main__":
    main()
