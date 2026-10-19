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
    modules = ("numpy", "scipy", "einops", "opt_einsum", "pandas", "matplotlib", "tqdm")
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "available")
    print(json.dumps(dict(status="cuda_ready", **cuda_environment(device), dependencies=versions)))

if __name__ == "__main__":
    main()
