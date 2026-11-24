"""사전 근거와 최근 대화 주제로 한국어 응답을 구성하는 규칙 기반 대화기."""

import re
from concurrent.futures import ThreadPoolExecutor

from dictionary import extract_query
from dialogue import daily_reply

PARTICLES = re.compile(
    r"(?:에게서|한테서|에서는|으로는|에게|한테|에서|처럼|하고|이랑|랑|와|과|으로|로|을|를|은|는|가|의|에|도|만)$"
)
STOPWORDS = set(
    "나는 내가 저는 제가 너는 네가 그게 이게 그거 이거 그건 이건 오늘 어제 정말 너무 조금 그냥 함께 대해 대해서 알려줘 설명해줘 이야기 얘기 싶어 있어 없어 어떤 무슨 왜 어떻게 해줘 할까 할지 뭐야 뭔가요 무엇인가요 차이 차이는 차이가 다른 다를까 차이점 비교 쉽게 더 자세히 예시 예를 들면 그러면 그럼 그리고 궁금해 생각해 생각이 들어".split()
)
SHORT_WORDS = set("돈 집 밥 꿈 삶 꽃 물 책 일 차 말 길 빛 밤 힘".split())
FOLLOWUP = re.compile(
    r"^(?:그럼\s*|그러면\s*)?(?:그거|그건|그게|그것|이거|그\s*말|더|조금\s*더|쉽게|자세히|예시|예를|다시)"
)


def clean_word(word):
    return re.sub(r"[-^\s\d]", "", word)


def keywords(text):
    """형태소 분석기 대신 보수적인 조사 제거로 최대 두 주제를 고른다."""
    result = []
    for token in re.findall(r"[가-힣A-Za-z]+", text):
        if token in STOPWORDS or (len(token) < 2 and token not in SHORT_WORDS):
            continue
        candidate = PARTICLES.sub("", token)
        word = candidate if len(token) > 2 or candidate in SHORT_WORDS else token
        if (
            word in STOPWORDS
            or (len(word) < 2 and word not in SHORT_WORDS)
            or word in result
        ):
            continue
        if word.endswith(("해줘", "했어", "해서", "할까", "싶어", "같아", "인가요")):
            continue
        result.append(word)
    return result[:2]


def recent_topics(messages):
    # 클라이언트가 돌려준 메타데이터도 길이·형식을 제한하고 마지막 답변만 본다.
    for message in reversed(messages[:-1]):
        if message.get("role") == "assistant":
            topics = message.get("topics", [])
            if isinstance(topics, list):
                return [
                    t
                    for t in topics[:2]
                    if isinstance(t, str) and re.fullmatch(r"[가-힣A-Za-z ]{1,80}", t)
                ]
            return []
    return []


def select_sense(entries, topic, context):
    candidates = [
        e for e in entries if clean_word(e["word"]) == clean_word(topic)
    ] or entries
    context_words = (
        (set(re.findall(r"[가-힣A-Za-z]{2,}", context)) | set(keywords(context)))
        - {topic}
        - STOPWORDS
    )

    def score(entry):
        definition = entry["definition"]
        overlap = sum(word in definition for word in context_words)
        return (
            overlap * 10
            + (1 if entry["dictionary"] == "표준국어대사전" else 0)
            - (3 if definition.startswith(("→", "⇒")) else 0)
        )

    return max(candidates, key=score) if candidates else None


def particle(word, consonant, vowel):
    last = ord(word[-1]) if word else 0
    return consonant if 0xAC00 <= last <= 0xD7A3 and (last - 0xAC00) % 28 else vowel


def conversational_definition(definition):
    # 뜻의 내용을 새로 추측하지 않고 종결 표현만 대화체로 바꾼다.
    sentences = re.split(r"(?<=[.!?])\s+", definition.strip())[:2]
    rewritten = []
    endings = {
        "하다": "해",
        "되다": "돼",
        "이다": "이야",
        "있다": "있어",
        "없다": "없어",
        "느끼다": "느껴",
        "여기다": "여겨",
        "이르다": "일러",
    }
    for sentence in sentences:
        sentence = sentence.rstrip(".。 ")
        if not sentence:
            continue
        if sentence.endswith("다"):
            for ending, replacement in endings.items():
                if sentence.endswith(ending):
                    sentence = sentence[: -len(ending)] + replacement
                    break
            else:
                # 활용 규칙을 모르는 서술은 원문 그대로 인용한다.
                sentence = f"“{sentence}”라는 뜻이야"
        else:
            sentence += particle(sentence, "이야", "야")
        rewritten.append(sentence + ".")
    return " ".join(rewritten)


def describe(entry):
    word = entry["word"]
    return f'“{word}”{particle(clean_word(word), "은", "는")} {conversational_definition(entry["definition"])}'


def comparison_insight(known):
    definitions = [e["definition"] for _, e in known]
    if all(re.search(r"마음|(?:^|\s)정(?:[. ]|$)", d) for d in definitions):
        friend = next((e for _, e in known if "친구 사이" in e["definition"]), None)
        if friend:
            return f'두 말 모두 마음이나 정을 이야기하지만, “{friend["word"]}”의 풀이에는 특히 ‘친구 사이’라는 관계가 명시돼 있어.'
    return "두 말은 위와 같은 의미로 풀이돼. 뜻의 차이를 실제 문장과 함께 보면 더 분명해질 수 있어."


def make_example(entry):
    definition = entry["definition"]
    if "친구" in definition:
        return "힘든 날 친구가 옆에서 이야기를 들어 주는 장면을 떠올려 볼 수 있어."
    if "아끼" in definition or "귀중" in definition:
        return (
            "소중한 사람이 지쳐 보일 때 안부를 묻고 챙겨 주는 장면을 떠올려 볼 수 있어."
        )
    if "식물" in definition and ("줄기" in definition or "가지" in definition):
        return "공원에서 줄기와 가지가 자란 나무를 보고 이야기하는 상황을 떠올려 봐."
    return None


def dictionary_requested(messages):
    prompt = messages[-1]["content"].strip()
    if re.search(
        r"뜻.*(?:말고|말하지|알려주지|필요\s*없)|뜻풀이.*(?:말고|하지)", prompt
    ):
        return False
    explicit = extract_query(prompt)
    if explicit and (explicit[1] or re.match(r"^(?:사전|단어)\s", prompt)):
        return True
    if explicit and re.search(r"뜻|의미|정의", prompt):
        return True
    if re.search(r"(?:뜻|의미|정의).*?(?:비교|차이|알려|설명)", prompt):
        return True
    previous = next(
        (m for m in reversed(messages[:-1]) if m["role"] == "assistant"), {}
    )
    return (
        previous.get("source") == "dictionary"
        and bool(FOLLOWUP.match(prompt))
        and bool(re.search(r"예시|예를|설명|풀어|알려", prompt))
    )


class ConversationEngine:
    def __init__(self, dictionaries):
        self.dictionaries = dictionaries

    def reply(self, messages, local_result):
        prompt = messages[-1]["content"].strip()
        explicit = extract_query(prompt)
        declines_definition = bool(
            re.search(
                r"뜻.*(?:말고|말하지|알려주지|필요\s*없)|뜻풀이.*(?:말고|하지)", prompt
            )
        )
        if declines_definition:
            return daily_reply(messages, local_result)
        # 특정 사전 지명이나 명시적인 검색 명령은 원문 목록을 그대로 제공한다.
        if explicit and (explicit[1] or re.match(r"^(?:사전|단어)\s", prompt)):
            result = self.dictionaries.lookup(*explicit)
            result["topics"] = [explicit[0]]
            return result
        previous = next(
            (m for m in reversed(messages[:-1]) if m["role"] == "assistant"), {}
        )
        definition_request = bool(explicit and re.search(r"뜻|의미|정의", prompt))
        definition_request = definition_request or bool(
            re.search(r"(?:뜻|의미|정의).*?(?:비교|차이|알려|설명)", prompt)
        )
        definition_followup = (
            previous.get("mode") == "dictionary"
            and bool(FOLLOWUP.match(prompt))
            and bool(re.search(r"예시|예를|설명|풀어|알려", prompt))
        )
        if not definition_request and not definition_followup:
            return daily_reply(messages, local_result)
        comparison = bool(re.search(r"차이|비교|다를까|달라", prompt))
        followup = bool(FOLLOWUP.match(prompt)) or bool(
            re.fullmatch(
                r"(?:그럼\s*)?(?:왜|어떻게)(?:\s*(?:그래|해야\s*해|되는\s*거야))?[?!. ]*",
                prompt,
            )
        )
        topics = recent_topics(messages) if followup else []
        if explicit:
            topics = [explicit[0]]
        elif comparison:
            pair = re.search(
                r"(.+?)(?:와|과|하고|이랑|랑)\s+(.+?)(?:의\s*)?(?:차이|비교|뭐가\s*달라|어떻게\s*달라)",
                prompt,
            )
            topics = (
                [
                    keywords(p)[0] if keywords(p) else p.strip(" ?")
                    for p in pair.groups()
                ]
                if pair
                else keywords(prompt)
            )
        elif followup and not topics:
            return {
                **local_result,
                "content": "어떤 말이나 주제를 이어서 설명하면 좋을까? 먼저 궁금한 주제를 알려줘.",
            }
        elif not topics:
            topics = keywords(prompt)
        topics = list(dict.fromkeys(t for t in topics if t and len(t) <= 80))[:2]
        if not topics:
            return local_result
        if not self.dictionaries.enabled and not explicit:
            return local_result
        with ThreadPoolExecutor(max_workers=2) as executor:
            lookups = list(executor.map(self.dictionaries.lookup, topics))
        context = (
            " ".join(
                message["content"]
                for message in messages[-5:]
                if message["role"] == "user"
            )
            if followup or definition_request
            else prompt
        )
        selected = [
            select_sense(result.get("sources", []), topic, context)
            for topic, result in zip(topics, lookups)
        ]
        known = [(topic, entry) for topic, entry in zip(topics, selected) if entry]
        if not known:
            if explicit:
                return lookups[0]
            return {
                **local_result,
                "content": f"“{topics[0]}”의 뜻을 확인하지 못했어. 단어를 다시 확인해 줄래?",
                "topics": topics,
            }
        sources = []
        for result in lookups:
            for entry in result.get("sources", []):
                if entry in selected and entry not in sources:
                    sources.append(entry)
        facts = "\n".join(describe(entry) for _, entry in known)
        if comparison and len(known) == 2:
            content = f"내가 찾은 뜻을 바탕으로 두 말을 비교해 볼게.\n\n{facts}\n\n{comparison_insight(known)} 어느 상황에서 두 말을 쓰려는 거야? 문장이 있으면 그 맥락에 맞는 의미를 골라볼 수 있어."
        elif comparison:
            missing = [topic for topic, entry in zip(topics, selected) if not entry]
            content = f"{facts}\n\n" + (
                f"다만 “{missing[0]}”의 뜻은 확인하지 못해서 두 말의 차이를 단정할 수 없어."
                if missing
                else "비교할 다른 단어도 알려줄래?"
            )
        elif followup:
            content = f"앞에서 이야기한 “{topics[0]}”{particle(topics[0], '을', '를')} 이어서 말하면,\n\n{facts}\n\n"
            if re.search(r"예시|예를", prompt):
                examples = [(entry["word"], make_example(entry)) for _, entry in known]
                examples = [(word, example) for word, example in examples if example]
                content += (
                    "뜻을 참고해 내가 만든 예를 들면,\n"
                    + "\n".join(f"· {word}: {example}" for word, example in examples)
                    + "\n실제 사전 용례를 인용한 건 아니야. 너도 비슷한 경험이 있어?"
                    if examples
                    else "사전 뜻만으로 구체적인 상황 예시를 만들기에는 근거가 부족해. 네가 이 말을 쓰려는 문장을 알려줄래?"
                )
            elif re.search(r"왜|어떻게", prompt):
                content += "사전 뜻만으로 원인이나 방법까지 설명할 수는 없어. 어떤 일이 있었는지 알려주면 그 상황부터 함께 정리해 보자."
            else:
                content += "이 뜻 가운데 어떤 부분이 어렵거나 더 궁금해? 네가 떠올린 상황과 연결해서 이야기해 보자."
        elif explicit:
            content = f"{facts}\n\n너는 이 말을 어떤 상황에서 봤어? 그 맥락이 있으면 여러 뜻 중 더 어울리는 풀이를 찾아볼 수 있어."
        else:
            content = f"“{topics[0]}”에 관한 이야기구나.\n\n{facts}\n\n"
            if re.search(r"왜|어떻게|방법", prompt):
                content += "뜻은 확인했지만 그것만으로 원인이나 방법을 단정하기는 어려워. 어떤 상황인지 조금 더 이야기해 줄래?"
            else:
                content += f"너는 “{topics[0]}”에 대해 어떤 생각이 들어? 경험이나 궁금한 점을 편하게 들려줘."
        notices = []
        for result in lookups:
            # 실패한 사전이 있어도 얻은 근거는 사용한다. 오류는 답변 끝에 남긴다.
            for line in result["content"].splitlines():
                if (
                    re.match(r"^(?:표준국어대사전|우리말샘):", line)
                    and "일치하는 표제어" not in line
                    and line not in notices
                ):
                    notices.append(line)
        if notices:
            content += "\n\n" + "\n".join(notices)
        return {
            "content": content,
            "mode": "dictionary",
            "source": "dictionary",
            "style": "conversation",
            "sources": sources,
            "topics": topics,
        }
