import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd

from test_data import transactions

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_llm_identity_changes_when_cached_data_changes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("runner", ROOT / "run_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        base = ROOT / "data/test_artifacts" / uuid.uuid4().hex
        (base / "basic").mkdir(parents=True)
        csv = base / "basic/llm_seed0.csv"
        csv.write_text("value\n1\n", encoding="utf-8")
        meta = {"model": "test-model", "seed": 0, "variant": "basic", "prompt_version": "paired-basic-aml-v1"}
        csv.with_name("llm_seed0_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        first = runner.llm_cache_identity(base, ["E3-basic"], [0])
        csv.write_text("value\n2\n", encoding="utf-8")
        second = runner.llm_cache_identity(base, ["E3-basic"], [0])
        self.assertNotEqual(first, second)

    def test_prepare_exports_only_training_positives_and_versioned_manifest(self):
        # Use normal directory permissions: Windows sandbox denies chmod(0700)
        # directories created by tempfile.TemporaryDirectory.
        base = ROOT / "data/test_artifacts" / uuid.uuid4().hex
        base.mkdir(parents=True)
        df = pd.concat([transactions()] * 4, ignore_index=True)
        df["Timestamp"] = pd.date_range("2022-09-01", periods=12, freq="min").strftime("%Y/%m/%d %H:%M")
        raw = base / "raw.csv"
        df.to_csv(raw, index=False)
        out = base / "results"
        env = dict(os.environ, AML_RAW_CSV=str(raw), AML_FEATURES_PARQUET=str(base / "features.parquet"))
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run_experiments.py"),
                                 "--prepare-only", "--output-dir", str(out), "--rebuild-features"],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["splits"]["train"]["rows"], 7)
        self.assertEqual(manifest["splits"]["train"]["positives"], 2)
        self.assertEqual(len(pd.read_csv(out / "train_positives.csv")), 2)
        self.assertEqual(manifest["feature_version"], "causal-seconds-v3")
        self.assertNotIn("llm_config.py", manifest["source_sha256"])

        # A changed raw source must not silently reuse features from the old one.
        df.loc[0, "Amount Paid"] = 9999.
        df.to_csv(raw, index=False)
        changed = subprocess.run([sys.executable, "-B", str(ROOT / "run_experiments.py"),
                                  "--prepare-only", "--output-dir", str(base / "changed")],
                                 cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(changed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
