"""학습 언어와 형태소 경계 정책을 비교하는 손실 없는 BPE 예비 실험.

이 토크나이저는 Qwen의 임베딩/출력층과 호환되지 않는다.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from korean_tokenizer import LOCK, analyzer

PILOT = Path(__file__).parent / "research" / "pilot_corpus.json"


def pilot_data():
    return json.loads(PILOT.read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def morphology_spans(text):
    """원형으로 치환하지 않고 원문의 경계에서만 자른다. 공백도 보존한다."""
    with LOCK:
        tokens = analyzer().tokenize(text)
    boundaries = sorted(
        {0, len(text)} | {p for t in tokens for p in (t.start, t.start + t.len)}
    )
    return [text[a:b] for a, b in zip(boundaries, boundaries[1:]) if a < b]


def training_prefix(text, budget):
    # 잘린 UTF-8 문자의 마지막 1~3바이트만 제외. 실제 바이트 수를 따로 기록한다.
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def train(text, vocab_size, morphology=False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        morphology_spans(text) if morphology else [text],
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tokenizer


def encode(tokenizer, text, morphology=False):
    chunks = morphology_spans(text) if morphology else [text]
    return [
        i
        for chunk in chunks
        for i in tokenizer.encode(chunk, add_special_tokens=False).ids
    ]


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("JSON 객체가 필요합니다.")
    for key in ("train_ko", "train_en"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError("한국어와 영어 학습 자료가 모두 필요합니다.")
        if len(data[key].encode()) > 30000:
            raise ValueError("학습 자료는 언어별 30KB 이하로 입력해 주세요.")
    pairs = data.get("test_pairs")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 30:
        raise ValueError("평가 문장 쌍을 1~30개 입력해 주세요.")
    for pair in pairs:
        if not isinstance(pair, dict) or any(
            not isinstance(pair.get(lang), str)
            or not pair[lang].strip()
            or len(pair[lang]) > 1000
            for lang in ("ko", "en")
        ):
            raise ValueError("각 평가 쌍에 한국어와 영어 문장이 필요합니다.")
        for lang in ("ko", "en"):
            # 동일 문장/부분 문자열 누출 방지. 유사 문장까지 독립임을 보장하지는 않는다.
            if pair[lang].strip() in data[f"train_{lang}"]:
                raise ValueError("평가 문장이 학습 자료에 포함되어 있습니다.")
    vocab = data.get("vocab_size", 512)
    budget = data.get("byte_budget", 1000)
    if type(vocab) is not int or not 256 <= vocab <= 4096:
        raise ValueError("어휘 크기는 256~4096 정수여야 합니다.")
    if type(budget) is not int or not 256 <= budget <= 30000:
        raise ValueError("학습 바이트 한도는 256~30000 정수여야 합니다.")
    return vocab, budget


def experiment(data):
    vocab_size, budget = validate(data)
    corpora = {
        lang: training_prefix(data[f"train_{lang}"], budget) for lang in ("ko", "en")
    }
    conditions, rows, artifacts = [], [], {}
    for name, lang, morphology in (
        ("english_bpe", "en", False),
        ("korean_bpe", "ko", False),
        ("korean_morph_bpe", "ko", True),
    ):
        tokenizer = train(corpora[lang], vocab_size, morphology)
        artifact = tokenizer.to_str()
        artifacts[name] = {
            "tokenizer_json": json.loads(artifact),
            "morphology_boundaries": morphology,
        }
        condition = {
            "name": name,
            "training_language": lang,
            "morphology_boundaries": morphology,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": digest(artifact),
        }
        totals = {"ko": 0, "en": 0}
        for index, pair in enumerate(data["test_pairs"]):
            for language in ("ko", "en"):
                text = pair[language]
                ids = encode(tokenizer, text, morphology)
                restored = tokenizer.decode(ids, skip_special_tokens=False)
                if restored != text:
                    raise RuntimeError("토큰의 원문 복원 검증에 실패했습니다.")
                totals[language] += len(ids)
                rows.append(
                    {
                        "condition": name,
                        "pair": index + 1,
                        "language": language,
                        "text": text,
                        "ids": ids,
                        "pieces": [tokenizer.id_to_token(i) for i in ids],
                        "token_count": len(ids),
                        "characters": len(text),
                        "utf8_bytes": len(text.encode()),
                        "tokens_per_character": len(ids) / len(text),
                        "roundtrip": restored == text,
                    }
                )
        condition.update(
            {
                "ko_tokens": totals["ko"],
                "en_tokens": totals["en"],
                "ko_en_ratio": totals["ko"] / totals["en"],
            }
        )
        conditions.append(condition)
    sizes = [c["actual_vocab_size"] for c in conditions]
    warnings = [
        "토큰 수는 대화 품질·이해력·모델 정확도를 측정하지 않습니다.",
        "이 연구용 토크나이저는 현재 Qwen 채팅 모델에 연결되어 있지 않습니다.",
        "GPT-5.6 관련 도구라는 사용자 기억만 기록했습니다. 이전 도구의 URL·인코딩·입력 자료는 미확인입니다.",
        "ByteLevel BPE는 GPT 토크나이저의 정확한 재현이 아닙니다. GPT 수치는 이 실험에 포함하지 않았습니다.",
        "예산은 UTF-8 바이트 상한입니다. 문자가 잘리면 끝 바이트를 제외하므로 실제 학습량이 조금 다를 수 있습니다.",
        "원문 형태소 경계는 Kiwi의 시작·끝 위치이며, 활용형의 중첩 경계도 분할합니다. 언어학적 정답 경계를 보장하지 않습니다.",
    ]
    if len(set(sizes)) != 1 or any(size != vocab_size for size in sizes):
        warnings.append(
            "자료 부족 또는 경계 제약으로 목표 어휘 크기에 도달하지 못했습니다. 동일 어휘 예산의 확정 비교로 해석하지 마세요."
        )
    if any(len(text.encode()) < budget - 3 for text in corpora.values()):
        warnings.append(
            "일부 언어의 학습 자료가 바이트 한도보다 적습니다. 자료량이 통제되지 않았습니다."
        )
    return {
        "schema_version": 1,
        "purpose": "은수 한국어 토큰화 예비 실험 · 언어 모델 학습 전",
        "provenance": data.get(
            "provenance", "사용자 입력 · 출처와 번역 품질은 별도 확인 필요"
        ),
        "settings": {
            "target_vocab_size": vocab_size,
            "byte_budget": budget,
            "normalization": "none",
            "byte_alphabet": 256,
            "min_frequency": 1,
            "special_tokens": [],
            "add_prefix_space": False,
            "versions": {
                p: importlib.metadata.version(p) for p in ("tokenizers", "kiwipiepy")
            },
        },
        "training": {
            lang: {
                "text": text,
                "utf8_bytes": len(text.encode()),
                "sha256": digest(text),
            }
            for lang, text in corpora.items()
        },
        "evaluation_sha256": digest(
            json.dumps(data["test_pairs"], ensure_ascii=False, sort_keys=True)
        ),
        "conditions": conditions,
        "rows": rows,
        "artifacts": artifacts,
        "warnings": warnings,
    }
