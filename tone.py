"""Kiwi의 활용 규칙으로 답변의 종결 어미와 높임 표현을 일관되게 맞춘다."""

import re
from korean_tokenizer import analyzer, LOCK


def polite_requested(messages):
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        text = message["content"]
        if re.search(
            r"(?:존댓말|높임말|정중한\s*말투)(?:로|을|\s)*(?:말|답|해|써|부탁)|존댓말\s*(?:해|써)",
            text,
        ):
            return True
        if re.search(r"반말(?:로|을|\s)*(?:말|답|해|써|부탁)", text):
            return False
    return False


def normalize_tone(text, polite=False):
    lines, code = [], False
    with LOCK:
        kiwi = analyzer()
        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                code = not code
                lines.append(line)
                continue
            if code or not line.strip() or "`" in line:
                lines.append(line)
                continue
            tokens = kiwi.tokenize(line)
            pairs, changed = [], False
            for token in tokens:
                form, tag = token.form, token.tag
                if not polite and tag == "EP" and form in ("시", "으시"):
                    changed = True
                    continue
                replacement = form
                if not polite:
                    if tag.startswith(("VV", "VX")) and form == "드리":
                        replacement = "주"
                    if tag == "NP" and form == "저":
                        replacement = "나"
                    if tag == "EF":
                        if form in ("습니다", "ᆸ니다", "ㅂ니다"):
                            replacement = (
                                "야"
                                if pairs and pairs[-1][1] in ("VCP", "VCN")
                                else "어"
                            )
                        elif form in ("에요", "예요"):
                            replacement = "야"
                        elif form in ("나요", "습니까", "ᆸ니까", "ㅂ니까"):
                            replacement = "어"
                        elif form == "죠":
                            replacement = "지"
                        elif form.endswith("요"):
                            replacement = form[:-1]
                elif tag == "EF":
                    replacement = {
                        "어": "어요",
                        "아": "아요",
                        "야": "에요",
                        "네": "네요",
                        "지": "죠",
                        "니": "나요",
                        "냐": "나요",
                        "다": "어요",
                    }.get(form, form)
                    if form.endswith(("까", "게")):
                        replacement = form + "요"
                changed = changed or replacement != form
                pairs.append((replacement, tag))
            lines.append(kiwi.join(pairs) if changed else line)
    result = "\n".join(lines)
    # 짧은 잡담에서 같은 역할의 질문만 여러 개 이어질 때 마지막 하나를 남긴다.
    questions = re.split(r"(?<=[?？])\s+", result.strip())
    if (
        len(questions) > 1
        and all(part.endswith(("?", "？")) for part in questions)
        and not code
    ):
        result = questions[-1]
    return result
