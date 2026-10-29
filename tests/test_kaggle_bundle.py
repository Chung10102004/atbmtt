import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_kernel", ROOT / "kaggle/build_kernel.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class KaggleBundleTests(unittest.TestCase):
    def test_bundle_has_portable_paths_and_excludes_secrets_and_old_caches(self):
        files = builder.bundle_files()
        self.assertIn("src/data.py", files)
        self.assertIn("run_experiments.py", files)
        self.assertIn("src/llm_validate.py", files)
        self.assertNotIn("llm_config.py", files)
        self.assertFalse(any("\\" in name for name in files))
        self.assertFalse(any(name.startswith("data/") for name in files))

    def test_kernel_is_valid_python_and_private_with_explicit_owner(self):
        source, meta = builder.build("example-user", "aml-rerun", ["--exps", "E0", "ER"])
        compile(source, "kernel.py", "exec")
        self.assertEqual(meta["id"], "example-user/aml-rerun")
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])

    def test_path_traversal_in_cache_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            builder.bundle_files(llm_dir=ROOT.parent)


if __name__ == "__main__":
    unittest.main()
