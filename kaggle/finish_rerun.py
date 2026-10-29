"""Wait for complete LLM caches, run their private Kaggle benchmark, and download/report.

This controller does not call the LLM API. Generation has its own persistent budget.
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def cache_progress(directory, n):
    total, ready = 0, True
    for variant in ("basic", "aml"):
        for seed in (0, 1, 2):
            path = directory / variant / f"llm_seed{seed}_meta.json"
            if not path.exists():
                ready = False
                continue
            meta = json.loads(path.read_text(encoding="utf-8"))
            total += meta.get("rows", 0)
            ready &= meta.get("complete", False) and meta.get("rows") == n
    return bool(ready), total


def check_core_equivalence():
    snapshot = ROOT / "results/rerun_kaggle_baselines/submission_v1/kernel.py"
    tree = ast.parse(snapshot.read_text(encoding="utf-8"))
    files = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "FILES" for target in node.targets))
    checked = {}
    for name, functions in {
        "src/data.py": ["load_raw", "build_features", "_account_history_features", "temporal_split"],
        "src/model.py": ["train", "evaluate", "best_f1_threshold", "recall_at_fpr", "alert_budget_metrics"],
        "src/augment.py": ["_postprocess", "gen_random", "gen_smote", "gen_ctgan", "synth_quality"],
        "run_experiments.py": ["run_one"],
    }.items():
        def bodies(source):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        before, after = bodies(files[name]), bodies((ROOT / name).read_text(encoding="utf-8"))
        for function in functions:
            if before[function] != after[function]:
                raise ValueError(f"Core logic changed since baseline: {name}:{function}")
        checked[name] = functions
    return checked


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", default="aml-llm-rerun-20261029")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--max-wait-seconds", type=int, default=7200)
    args = ap.parse_args()
    output = ROOT / "results/rerun_analysis"
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "pipeline.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    log = output / "pipeline.log"
    def message(text):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"{stamp} {text}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    def command(parts):
        result = subprocess.run([sys.executable, "-B", *parts], cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace")
        if result.stdout.strip():
            message(result.stdout.strip())
        if result.stderr.strip():
            message(result.stderr.strip())
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {parts[0]}")
        return result.stdout
    try:
        base = ROOT / "data/llm/rerun_20261029"
        positives = ROOT / "results/rerun_kaggle_baselines/results/rerun_20261029/train_positives.csv"
        import pandas as pd
        x = pd.read_csv(positives)
        n = round(.5 * len(x))
        deadline = time.monotonic() + args.max_wait_seconds
        while True:
            ready, rows = cache_progress(base, n)
            message(f"LLM generation: {rows}/{n * 6} rows; ready={ready}")
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Generation wait exceeded limit; inspect generator logs")
            time.sleep(max(1, min(60, args.poll_seconds)))
        from augment import gen_llm_variant
        from llm_budget import BudgetLedger
        os.environ["AML_LLM_DIR"] = str(base)
        ledger = BudgetLedger(base / "budget.sqlite", 2., .1, .5)
        if ledger.total() > 2. + 1e-10:
            raise ValueError("Committed API cost exceeds the agreed cap")
        for variant in ("basic", "aml"):
            for seed in (0, 1, 2):
                gen_llm_variant(x, n, seed, variant)
        equivalence = check_core_equivalence()
        (output / "baseline_core_equivalence.json").write_text(json.dumps(equivalence, indent=2), encoding="utf-8")
        state_path = output / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        kernel = f"{args.owner}/{args.slug}"
        if not state.get("submitted"):
            command(["kaggle/build_kernel.py", "--owner", args.owner, "--slug", args.slug,
                     "--output-dir", "kaggle/kernel_llm", "--llm-dir", str(base), "--args",
                     "--exps E3-basic E3-AML --seeds 0 1 2 --ratio 0.5 --rebuild-features --output-dir results/rerun_llm_20261029"])
            snapshot = ROOT / "results/rerun_kaggle_llm/submission_v1"
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in ("kernel.py", "kernel-metadata.json"):
                shutil.copyfile(ROOT / "kaggle/kernel_llm" / name, snapshot / name)
            command(["kaggle/run_cli.py", "kernels", "push", "-p", "kaggle/kernel_llm",
                     "--accelerator", "NvidiaTeslaT4", "--timeout", "10800"])
            state = {"submitted": True, "kernel": kernel}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        while True:
            status = command(["kaggle/run_cli.py", "kernels", "status", kernel])
            if "COMPLETE" in status:
                break
            if any(word in status for word in ("ERROR", "FAILED", "CANCELLED")):
                command(["kaggle/run_cli.py", "kernels", "logs", kernel])
                raise RuntimeError("Kaggle LLM benchmark failed; inspect pipeline.log")
            if time.monotonic() > deadline + 3600:
                raise TimeoutError("Kaggle execution wait exceeded limit")
            time.sleep(max(1, min(60, args.poll_seconds)))
        command(["kaggle/run_cli.py", "kernels", "output", kernel, "-p", "results/rerun_kaggle_llm",
                 "--file-pattern", "^results/", "--quiet"])
        command(["analyze_results.py", "--result-dirs", "results/rerun_kaggle_baselines/results/rerun_20261029",
                 "results/rerun_kaggle_llm/results/rerun_llm_20261029", "--output-dir", str(output)])
        state["complete"] = True
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        message("COMPLETE: all 18 runs collected and final report/figures written")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
