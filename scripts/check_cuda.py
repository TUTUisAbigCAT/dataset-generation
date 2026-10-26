"""Check CUDA and required imports before a cluster run; never falls back."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cuda_runtime import require_cuda, cuda_environment

def main():
    device = require_cuda()
    from scripts.check_s4_backends import run_checks
    modules = ("numpy", "scipy", "einops", "opt_einsum", "pandas", "matplotlib", "tqdm")
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "available")
    s4_checks = run_checks("cuda", cuda_index=device.index)
    print(json.dumps(dict(status="cuda_ready", **cuda_environment(device),
                         dependencies=versions, s4_complex_checks=s4_checks)))

if __name__ == "__main__":
    main()
