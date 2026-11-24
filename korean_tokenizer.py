"""한국어 형태소 분석과 Qwen3의 실제 BPE 인코딩을 구분해 제공한다."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent
LOCK = Lock()


@lru_cache(maxsize=1)
def analyzer():
    from kiwipiepy import Kiwi

    return Kiwi(num_workers=1)


@lru_cache(maxsize=2)
def model_tokenizer(model="qwen3"):
    from tokenizers import Tokenizer

    path = ROOT / ".runtime" / "qwen3-tokenizer.json"
    if not path.is_file():
        raise RuntimeError(
            "모델 토크나이저 파일이 없어요. .venv/bin/python scripts/setup_tokenizer.py를 실행해 주세요."
        )
    return Tokenizer.from_file(str(path))


@lru_cache(maxsize=1)
def byte_decoder():
    """ByteLevel의 표시용 유니코드 기호를 원래 바이트 값으로 되돌린다."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = list(values)
    next_symbol = 256
    for value in range(256):
        if value not in values:
            values.append(value)
            alphabet.append(next_symbol)
            next_symbol += 1
    return {chr(symbol): value for symbol, value in zip(alphabet, values)}


def describe_model_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoder = byte_decoder()
    special_ids = {
        i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special
    }
    tokens = []
    for token_id, piece, (start, end) in zip(
        encoded.ids, encoded.tokens, encoded.offsets
    ):
        special = token_id in special_ids
        raw = None if special else bytes(decoder[char] for char in piece)
        try:
            display = piece if special else raw.decode("utf-8")
            fragment = False
        except UnicodeDecodeError:
            display = tokenizer.decode([token_id], skip_special_tokens=False)
            fragment = True
        tokens.append(
            {
                "id": token_id,
                "piece": piece,
                "text": display,
                "surface": text[start:end],
                "start": start,
                "end": end,
                "bytes": len(raw) if raw is not None else len(piece.encode()),
                "byte_hex": raw.hex(" ").upper() if raw is not None else None,
                "byte_fragment": fragment,
                "special": special,
            }
        )
    return tokens


def tokenize(text, mode="morpheme", model="qwen3"):
    if mode not in ("morpheme", "model"):
        raise ValueError("지원하지 않는 분석 모드입니다.")
    if len(text) > 10000:
        raise ValueError("분석할 텍스트는 10,000자 이하로 입력해 주세요.")
    if mode == "morpheme":
        with LOCK:
            analyzed = analyzer().tokenize(text)
        tokens = [
            {
                "text": t.form,
                "surface": text[t.start : t.start + t.len],
                "pos": t.tag,
                "start": t.start,
                "end": t.start + t.len,
                "bytes": len(t.form.encode("utf-8")),
            }
            for t in analyzed
        ]
        method = "Kiwi 한국어 형태소 분석"
    else:
        tokenizer = model_tokenizer(model)
        tokens = describe_model_tokens(text, tokenizer)
        method = f"{model} BPE (본문만, 채팅 템플릿 제외)"
    return {
        "tokens": tokens,
        "source_text": text,
        "characters": len(text),
        "bytes": len(text.encode("utf-8")),
        "method": method,
        "mode": mode,
        "scope": "body",
        "token_count": len(tokens),
        "fragment_tokens": sum(t.get("byte_fragment", False) for t in tokens),
    }
