import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import build_features
from test_data import transactions


class RerunGenerationTests(unittest.TestCase):
    def test_generated_cache_hash_survives_kaggle_text_bundle(self):
        spec = importlib.util.spec_from_file_location("generate_llm_rerun", ROOT / "generate_llm_rerun.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from llm_budget import BudgetLedger
        from test_kaggle_bundle import builder
        import hashlib
        output = ROOT / "data/llm/test_artifacts" / uuid.uuid4().hex
        output.mkdir(parents=True)
        x = build_features(transactions()).drop(columns=["timestamp", "is_laundering"])
        row = json.loads(x.iloc[[1]].to_json(orient="records"))[0]
        row["log_amt_paid"] = .5
        text = json.dumps({"rows": [row]})
        def response(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                                   usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=1000))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=response)))
        ledger = BudgetLedger(output / "budget.sqlite", 2., .1, .5)
        meta = module.generate_seed(client, "test-model", x, 1, 0, "basic", output, ledger, 2)
        key = (output / "basic/llm_seed0.csv").relative_to(ROOT).as_posix()
        bundled = builder.bundle_files(output)[key].encode("utf-8")
        self.assertEqual(hashlib.sha256(bundled).hexdigest(), meta["cache_sha256"])
        (output / "basic/llm_seed0.csv").write_text("interrupted checkpoint write", encoding="utf-8")
        module.generate_seed(client, "test-model", x, 1, 0, "basic", output, ledger, 2)
        self.assertEqual((output / "basic/llm_seed0.csv").read_bytes(), bundled)
        self.assertEqual(ledger.summary()["calls_by_status"]["settled"]["calls"], 1)

    def test_paid_batch_records_usage_and_rejects_impossible_history(self):
        spec = importlib.util.spec_from_file_location("generate_llm_rerun", ROOT / "generate_llm_rerun.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from llm_budget import BudgetLedger
        directory = ROOT / "data/test_artifacts" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        x = build_features(transactions()).drop(columns=["timestamp", "is_laundering"])
        row = json.loads(x.iloc[[1]].to_json(orient="records"))[0]
        row["log_amt_paid"] = .5
        bad = dict(row, src_prev_out=1.5)
        text = json.dumps({"rows": [row, bad]})

        def response(**kwargs):
            self.assertGreater(kwargs.get("max_completion_tokens", 0), 0)
            self.assertLessEqual(kwargs["max_completion_tokens"], 9000)
            self.assertNotIn("max_tokens", kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                                   usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=1000))

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=response)))
        ledger = BudgetLedger(directory / "budget.sqlite", 2., .1, .5)
        rows, stats = module.request_batch(client, "test-model", "prompt", x.columns, 0, ledger, directory, logical_id="batch0")
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["rejected_rows"], 1)
        self.assertAlmostEqual(ledger.total(), .0006)
        self.assertTrue(any(directory.glob("call_*_response.json")))
        replay, _ = module.request_batch(client, "test-model", "prompt", x.columns, 0, ledger, directory, logical_id="batch0")
        self.assertEqual(len(replay), 1)
        self.assertEqual(ledger.summary()["calls_by_status"]["settled"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
