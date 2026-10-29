import importlib.util
import json
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FinishTests(unittest.TestCase):
    def test_incomplete_generation_never_triggers_gpu_submission(self):
        spec = importlib.util.spec_from_file_location("finish", ROOT / "kaggle/finish_rerun.py")
        finish = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finish)
        base = ROOT / "data/test_artifacts" / uuid.uuid4().hex
        (base / "basic").mkdir(parents=True)
        meta = base / "basic/llm_seed0_meta.json"
        meta.write_text(json.dumps({"rows": 100, "complete": True}), encoding="utf-8")
        ready, rows = finish.cache_progress(base, 1148)
        self.assertFalse(ready)
        self.assertEqual(rows, 100)


if __name__ == "__main__":
    unittest.main()
