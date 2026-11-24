"""추가 패키지 없이 프로젝트의 .env 설정을 읽는다."""

import os
import re
import shlex
from pathlib import Path


def load_env_file(path):
    path = Path(path)
    if not path.is_file():
        return

    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f".env {number}번째 줄은 KEY=값 형식으로 작성해 주세요.")
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(
                f".env {number}번째 줄의 따옴표를 확인해 주세요."
            ) from None
        if len(parts) > 1:
            raise ValueError(
                f".env {number}번째 줄의 공백이 있는 값은 따옴표로 감싸 주세요."
            )
        # 터미널에서 명시적으로 지정한 환경변수를 우선한다.
        os.environ.setdefault(key, parts[0] if parts else "")
