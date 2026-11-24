#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import kiwipiepy, tokenizers" 2>/dev/null; then
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python scripts/setup_tokenizer.py
exec .venv/bin/python app.py
