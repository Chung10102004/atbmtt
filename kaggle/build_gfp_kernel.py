"""Private Kaggle script for the GFP study (run_gfp_study.py); no credentials bundled.

    python kaggle/build_gfp_kernel.py --owner hoangducchung
    kaggle kernels push -p kaggle/kernel_gfp
    kaggle kernels output hoangducchung/aml-gfp-study -p results/gfp_kaggle
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ["src/data.py", "src/model.py", "src/feature_study.py", "src/gfp_features.py", "src/typology.py",
             "run_gfp_study.py"]
# Main (Kaggle, Python 3.13) environment: same pins as the feature study
DEPENDENCIES = ["numpy==2.2.6", "pandas==2.3.3", "scikit-learn==1.7.2", "xgboost==3.1.1", "numba==0.61.2",
                "pyarrow==21.0.0", "tabulate"]
# snapml has no Python 3.13 wheel -> separate Python 3.12 environment just for GFP
GFP_DEPENDENCIES = ["snapml==1.17.2", "numpy==2.2.6", "pandas==2.3.3"]


def build(owner, slug):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner) or not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("Invalid owner or slug")
    files = {p: (ROOT / p).read_text(encoding="utf-8") for p in ALLOWLIST}
    source = f'''import os, subprocess, sys
from pathlib import Path
run = lambda *a, **k: subprocess.run([str(x) for x in a], check=True, **k)
run(sys.executable, "-m", "pip", "install", "-q", *{DEPENDENCIES!r})
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/tmp")
FILES = {files!r}
for rel, content in FILES.items():
    path = WORK / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
print(gpu.stdout, flush=True)
if gpu.returncode:
    raise RuntimeError("Requested GPU unavailable")
raw = next(Path("/kaggle/input").rglob("HI-Small_Trans.csv"))
patterns = next(Path("/kaggle/input").rglob("HI-Small_Patterns.txt"))

# 1) GFP features in a Python 3.12 env (written outside /kaggle/working: ~1GB, not kept as output)
run(sys.executable, "-m", "pip", "install", "-q", "uv")
env312 = TMP / "gfp312"
run(sys.executable, "-m", "uv", "venv", env312, "--python", "3.12")
run(sys.executable, "-m", "uv", "pip", "install", "--python", env312 / "bin/python", *{GFP_DEPENDENCIES!r})
gfp = TMP / "gfp_features.npz"
run(env312 / "bin/python", "-B", WORK / "src/gfp_features.py", "--raw", raw, "--out", gfp,
    env=dict(os.environ, PYTHONUNBUFFERED="1"))
run("cp", gfp.with_suffix(".json"), WORK / "gfp_features_meta.json")

# 2) Train/evaluate on GPU in the main env
env = dict(os.environ, AML_RAW_CSV=str(raw), AML_DEVICE="cuda", PYTHONUNBUFFERED="1")
run(sys.executable, "-B", WORK / "run_gfp_study.py", "--raw", raw, "--patterns", patterns, "--gfp", gfp,
    "--output-dir", "results/gfp_study", cwd=WORK, env=env)
'''
    compile(source, "kernel.py", "exec")
    metadata = {"id": f"{owner}/{slug}", "title": slug, "code_file": "kernel.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": ["ealtman2019/ibm-transactions-for-anti-money-laundering-aml"],
                "competition_sources": [], "kernel_sources": []}
    return source, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", default="aml-gfp-study")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "kaggle/kernel_gfp")
    args = ap.parse_args()
    source, meta = build(args.owner, args.slug)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kernel.py").write_text(source, encoding="utf-8")
    (args.output_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built private kernel {meta['id']}")


if __name__ == "__main__":
    main()
