#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${BASE_PYTHON:-$HOME/anaconda3/envs/py310/bin/python}"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv --system-site-packages .venv
fi
# Reuse the working CUDA 12.4 torch wheel. Never upgrade the shared conda env.
.venv/bin/python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 5 --timeout 30 \
  'transformers==4.44.2' 'tokenizers==0.19.1' 'huggingface-hub==0.36.0' \
  'datasets==3.6.0' 'sentencepiece==0.2.0' 'rouge-score==0.1.2' 'protobuf<5'
.venv/bin/python -u -c 'import torch,transformers; print("ENV", torch.__version__,transformers.__version__,"CUDA",torch.cuda.is_available()); assert torch.cuda.is_available()'
.venv/bin/python -m pip freeze > environment.freeze.txt
