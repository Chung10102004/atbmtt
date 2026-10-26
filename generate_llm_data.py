"""Call the ChatGPT API to generate synthetic laundering transactions (E3) and save them to data/llm/.

Run once (needs an API key in llm_config.py), then run E3 like the other experiments:
    python generate_llm_data.py --dry-run            # print a sample prompt, no API call
    python generate_llm_data.py --seeds 0 1 2        # generate ratio x (#positives in train) rows per seed
    python run_experiments.py --exps E3 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
import llm_config as cfg  # noqa: E402
from data import load_features, temporal_split  # noqa: E402
from llm_gen import LLM_DIR, build_prompt, cache_path, column_stats, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="number of rows to generate = ratio * number of real positives in train")
    ap.add_argument("--dry-run", action="store_true", help="only print the prompt, do not call the API")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    parts, _ = temporal_split(load_features())
    X_tr, y_tr = parts["train"]
    X_pos = X_tr[y_tr == 1].reset_index(drop=True)
    del parts, X_tr
    n = int(round(args.ratio * len(X_pos)))

    if args.dry_run:
        ex = X_pos.sample(cfg.EXAMPLES_PER_CALL, random_state=0)
        prompt = build_prompt(X_pos, ex, cfg.ROWS_PER_CALL, column_stats(X_pos))
        print(prompt)
        print(f"\n--- ~{len(prompt) // 4} prompt tokens/call, ~{int(np.ceil(n / cfg.ROWS_PER_CALL))} "
              f"calls/seed for {n} rows, model={cfg.OPENAI_MODEL}")
        return

    from openai import OpenAI
    client = OpenAI(api_key=cfg.get_api_key(), base_url=cfg.OPENAI_BASE_URL)
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        if cache_path(seed).exists() and not args.overwrite:
            print(f"seed {seed}: {cache_path(seed)} already exists, skipping (--overwrite to regenerate)")
            continue
        t0 = time.time()
        df, usage = generate(client, cfg.OPENAI_MODEL, X_pos, n, seed,
                             rows_per_call=cfg.ROWS_PER_CALL,
                             examples_per_call=cfg.EXAMPLES_PER_CALL,
                             temperature=cfg.TEMPERATURE, max_workers=cfg.MAX_WORKERS,
                             log_dir=LLM_DIR)
        df.to_csv(cache_path(seed), index=False)
        meta = {"seed": seed, "model": cfg.OPENAI_MODEL, "temperature": cfg.TEMPERATURE,
                "rows": len(df), "rows_per_call": cfg.ROWS_PER_CALL,
                "examples_per_call": cfg.EXAMPLES_PER_CALL, "secs": round(time.time() - t0, 1),
                "typology_counts": df["typology"].value_counts().to_dict(), **usage}
        (LLM_DIR / f"llm_seed{seed}_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
