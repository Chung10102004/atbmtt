"""Allowlisted private Kaggle script for the feature study; no credentials bundled."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ["src/data.py", "src/model.py", "src/feature_study.py", "run_feature_study.py",
             "tests/test_feature_study.py"]
DEPENDENCIES = ["numpy==2.2.6", "pandas==2.3.3", "scikit-learn==1.7.2",
                "xgboost==3.1.1", "numba==0.61.2", "pyarrow==21.0.0"]


def build(owner, slug):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",owner) or not re.fullmatch(r"[a-z0-9-]+",slug):
        raise ValueError("Invalid owner or slug")
    files = {p:(ROOT/p).read_text(encoding="utf-8") for p in ALLOWLIST}
    source = f'''import os, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","install","-q",*{DEPENDENCIES!r}],check=True)
WORK = Path("/kaggle/working")
FILES = {files!r}
for rel,content in FILES.items():
    path = WORK/rel
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding="utf-8")
gpu = subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True)
print(gpu.stdout,flush=True)
if gpu.returncode:
    raise RuntimeError("Requested GPU unavailable")
raw = next(Path("/kaggle/input").rglob("HI-Small_Trans.csv"))
env = dict(os.environ,AML_RAW_CSV=str(raw),AML_DEVICE="cuda",PYTHONUNBUFFERED="1")
subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p","test_feature_study.py","-v"],cwd=WORK,env=env,check=True)
subprocess.run([sys.executable,"-B",str(WORK/"run_feature_study.py"),"--output-dir","results/feature_study_20261029"],cwd=WORK,env=env,check=True)
'''
    compile(source,"kernel.py","exec")
    metadata = {"id":f"{owner}/{slug}","title":slug,"code_file":"kernel.py",
        "language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,
        "enable_internet":True,"dataset_sources":["ealtman2019/ibm-transactions-for-anti-money-laundering-aml"],
        "competition_sources":[],"kernel_sources":[]}
    return source, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--slug",default="aml-feature-study-20261029")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"kaggle/kernel_features")
    args = ap.parse_args()
    source,meta = build(args.owner,args.slug)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"kernel.py").write_text(source,encoding="utf-8")
    (args.output_dir/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Built private kernel {meta['id']}")


if __name__ == "__main__":
    main()
