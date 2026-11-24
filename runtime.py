"""프로젝트에 준비된 Ollama를 필요할 때 시작한다. 다른 서버는 종료하지 않는다."""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent


def ensure_local_runtime():
    if os.getenv("CHAT_ENGINE", "ollama") == "rules":
        return
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    # 주소 검증은 모델 클라이언트와 동일하게 적용한다.
    from local_model import LocalModel

    LocalModel(base_url=url)

    def ready():
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=1) as response:
                json.load(response)
            return True
        except (OSError, ValueError):
            return False

    if ready():
        return
    candidates = [ROOT / ".runtime/ollama/bin/ollama", ROOT / ".runtime/ollama/ollama"]
    binary = shutil.which("ollama") or next(
        (str(p) for p in candidates if p.is_file()), None
    )
    if binary is None:
        print("Ollama가 없어요. 설치한 후 ollama serve를 실행해 주세요.", flush=True)
        return
    (ROOT / ".runtime").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "OLLAMA_HOST": url.removeprefix("http://"),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    with (ROOT / ".runtime/ollama.log").open("a") as log:
        subprocess.Popen(
            [binary, "serve"], env=env, stdout=log, stderr=log, start_new_session=True
        )
    for _ in range(30):
        if ready():
            return
        time.sleep(0.3)
    print(
        "로컬 모델 서버가 아직 준비되지 않았어요. .runtime/ollama.log를 확인해 주세요.",
        flush=True,
    )
