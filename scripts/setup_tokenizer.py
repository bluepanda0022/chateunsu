"""공식 Qwen3 토크나이저를 로컬에 준비한다."""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dictionary import dictionary_ssl_context
from tokenizers import Tokenizer

path = ROOT / ".runtime" / "qwen3-tokenizer.json"
path.parent.mkdir(exist_ok=True)
if not path.is_file():
    url = (
        "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/tokenizer.json"
    )
    with urllib.request.urlopen(
        url, context=dictionary_ssl_context(), timeout=120
    ) as response:
        content = response.read()
    Tokenizer.from_str(content.decode())
    path.write_bytes(content)
    print("Qwen3 토크나이저 준비 완료")
