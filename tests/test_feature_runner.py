import sys
import unittest
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FeatureBundleTests(unittest.TestCase):
    def test_private_allowlisted_bundle(self):
        spec = importlib.util.spec_from_file_location("feature_builder", Path(__file__).resolve().parents[1]/"kaggle/build_feature_kernel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source, metadata = module.build("snnasnguyen", "aml-feature-study-20261029")
        compile(source, "kernel.py", "exec")
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertIn("run_feature_study.py", source)
        self.assertNotIn("llm_config.py", source)
        self.assertNotIn("load_dotenv", source)
        self.assertNotIn("\\\\", metadata["code_file"])


if __name__ == "__main__":
    unittest.main()
