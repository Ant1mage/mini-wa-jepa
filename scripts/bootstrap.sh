#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
if [ ! -d .deps/metadrive/.git ]; then
    git clone --single-branch https://github.com/metadriverse/metadrive.git .deps/metadrive
fi
git -C .deps/metadrive checkout fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1
.venv/bin/python -m pip install -r requirements-runtime.txt
.venv/bin/python -m pip install --no-deps -e .deps/metadrive
# Installing torch with its automatic dependencies would pull pytorch-triton-rocm.
# Eager execution and SDPA MATH have been verified with Triton absent.
.venv/bin/python -m pip install --no-deps torch==2.9.1+rocm6.4 --index-url https://download.pytorch.org/whl/rocm6.4
.venv/bin/python -m metadrive.pull_asset
echo 'Ready. Run: .venv/bin/python -m scripts.check_rocm'
