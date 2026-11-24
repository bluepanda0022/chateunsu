"""일상 대화의 질문 단계와 사용자 답변을 연결한다. 사전을 호출하지 않는다."""

import re

TOPICS = {"free", "day", "conflict", "feeling", "food", "hobby"}
STAGES = {
    "open",
    "day",
    "detail",
    "reason",
    "feeling",
    "support",
    "plan",
    "choice",
    "menu",
    "name",
    "experience",
    "listen",
    "done",
}


def previous_state(messages):
    for message in reversed(messages[:-1]):
        if message["role"] != "assistant":
            continue
        state = message.get("dialogue")
        if (
            isinstance(state, dict)
            and isinstance(state.get("topic"), str)
            and state.get("topic") in TOPICS
            and isinstance(state.get("stage"), str)
            and state.get("stage") in STAGES
        ):
            return {
                "topic": state["topic"],
                "stage": state["stage"],
                "detail": str(state.get("detail", ""))[:200],
            }
        # 저장된 이전 버전의 질문도 답변을 받을 수 있게 연결한다.
        text = message["content"]
        if "하루는 어땠어" in text:
            return {"topic": "day", "stage": "day", "detail": ""}
        if "밥이 당겨" in text:
            return {"topic": "food", "stage": "choice", "detail": ""}
        if "빠져 있는 게" in text:
            return {"topic": "hobby", "stage": "name", "detail": ""}
        return {"topic": "free", "stage": "open", "detail": ""}
    return {"topic": "free", "stage": "open", "detail": ""}


def respond(content, topic="free", stage="open", detail=""):
    return {
        "content": content,
        "source": "conversation",
        "mode": "local",
        "style": "chat",
        "sources": [],
        "topics": [],
        "dialogue": {"topic": topic, "stage": stage, "detail": detail[:200]},
    }


def short(text):
    return text.replace("\n", " ")[:100].rstrip(" .?!")


def daily_reply(messages, local_result):
    text = messages[-1]["content"].strip()
    key = re.sub(r"\s+|[.?!~]", "", text)
    negated = bool(re.search(r"않|아니|안\s", text))
    state = previous_state(messages)
    topic, stage, detail = state["topic"], state["stage"], state["detail"]
    yes = key in ("응", "웅", "네", "그래", "좋아", "맞아", "ㅇㅇ")
    no = key in ("아니", "아니야", "싫어", "별로", "ㄴㄴ")
    if local_result["source"] == "learned":
        return {**local_result, "dialogue": state}
    if re.search(r"다른\s*(?:얘기|이야기)|주제\s*바꾸|그만\s*(?:하자|얘기)", text):
        return respond("좋아, 다른 이야기 하자. 지금 하고 싶은 얘기가 있어?")
    if re.search(r"뜻.*말고|뜻풀이.*(?:말고|하지)", text):
        return respond(
            "알겠어. 뜻풀이는 네가 물을 때만 할게. 지금 어떤 이야기를 하고 싶어?"
        )
    if key in ("잘자", "잘가", "바이", "다음에봐", "나잘게"):
        return {
            **local_result,
            "dialogue": {"topic": "free", "stage": "open", "detail": ""},
        }
    if (
        key
        in (
            "고마워",
            "감사해",
            "감사합니다",
            "이름이뭐야",
            "너누구야",
            "자기소개해줘",
            "뭐해",
        )
        and local_result["source"] != "unknown"
    ):
        return {**local_result, "dialogue": state}
    if key in ("안녕", "안녕하세요", "하이", "안뇽", "반가워"):
        return {
            **respond("안녕! 오늘 하루는 어땠어?", "day", "day"),
            "source": local_result["source"],
        }
    if re.search(r"배고|뭐\s*먹|먹을까", text):
        return respond(
            "뭐 좀 먹고 싶구나. 든든한 밥이 좋아, 가벼운 간식이 좋아?", "food", "choice"
        )
    if re.search(r"심심|놀아줘", text):
        return respond(
            "나랑 수다 떨자. 요즘 즐겨 하는 게 있어? 게임이나 음악 같은 것도 좋아.",
            "hobby",
            "name",
        )

    # 새 상황이 제시되면 이전의 질문 단계보다 우선한다.
    if re.search(r"싸웠|다퉜|싸워|다투", text) and topic != "conflict":
        return respond(
            "누군가와 다퉜구나. 어떤 일로 그렇게 됐어?", "conflict", "reason", text
        )
    if topic == "conflict":
        if stage == "reason":
            if yes or no:
                return respond(
                    "무슨 일이 있었는지 아직 잘 모르겠어. 서로 어떤 말을 했는지부터 들려줄래?",
                    topic,
                    stage,
                    detail,
                )
            return respond(
                f"“{short(text)}”라는 일이 있었구나. 그때 너는 어떤 기분이었어?",
                topic,
                "feeling",
                text,
            )
        if stage == "feeling":
            prefix = (
                "그 일 때문에 속상했겠다."
                if re.search(r"속상|슬퍼|서운|화나|짜증", text) and not negated
                else "그때 네 마음은 그랬구나."
            )
            return respond(
                prefix
                + " 지금은 내가 그냥 들어 주면 좋겠어, 아니면 어떻게 말할지 같이 생각해 볼까?",
                topic,
                "support",
                detail,
            )
        if stage == "support":
            if re.search(r"듣|들어|그냥", text) or no:
                return respond(
                    "알겠어. 조언부터 하지 않고 들을게. 아직 못 한 말이 있으면 편하게 해 줘.",
                    topic,
                    "listen",
                    detail,
                )
            return respond(
                "상대의 의도를 단정하기보다, 있었던 일과 네 기분을 나눠서 말해 보는 건 어때? “그 말을 들었을 때 나는 속상했어”처럼. 네 말투로 한번 적어 볼래?",
                topic,
                "plan",
                detail,
            )
        if stage == "plan":
            if yes:
                return respond(
                    "좋아. 어떤 일이 있었고 어떤 기분이었는지 한 문장씩 적어 봐. 내가 듣고 같이 다듬어 줄게.",
                    topic,
                    stage,
                    detail,
                )
            if no:
                return respond(
                    "지금 당장 이야기할 필요는 없어. 잠깐 쉬면서 네 마음부터 정리해도 괜찮아.",
                    topic,
                    "listen",
                    detail,
                )
            return respond(
                f"“{short(text)}”라고 전하고 싶은 거구나. 네 기분을 담았네. 상대에게 바라는 점도 덧붙이고 싶어?",
                topic,
                "done",
                detail,
            )
        if stage == "listen":
            return respond(
                f"응, 듣고 있어. “{short(text)}”라는 부분이 마음에 남아 있구나. 더 하고 싶은 말이 있어?",
                topic,
                "listen",
                detail,
            )
        if yes:
            return respond(
                "그러면 상대가 어떻게 해 줬으면 좋겠는지도 말해 봐. 네가 바라는 걸 듣고 싶어.",
                topic,
                "plan",
                detail,
            )
        if no:
            return respond(
                "그럼 지금 정리한 마음까지만 담아도 되겠다. 이 이야기를 하고 나니 기분은 좀 어때?",
                topic,
                "done",
                detail,
            )

    if topic == "food":
        if stage == "choice":
            if re.search(r"밥|식사|든든", text):
                return respond(
                    "그럼 식사로 고르자. 매콤한 게 당겨, 담백한 게 당겨?",
                    topic,
                    "menu",
                    "식사",
                )
            if re.search(r"간식|가볍", text):
                return respond(
                    "간식 좋지. 달달한 맛이 좋아, 짭짤한 맛이 좋아?",
                    topic,
                    "menu",
                    "간식",
                )
            return respond(
                "밥이랑 간식 중 어느 쪽이 더 끌려? 먹고 싶은 음식 이름을 말해도 돼.",
                topic,
                "choice",
                detail,
            )
        if stage == "menu":
            menu = "김치볶음밥" if "매콤" in text else "달걀국과 밥"
            if detail == "간식":
                menu = "과일이나 요거트" if "달" in text else "크래커"
            return respond(
                f"그럼 {menu} 어때? 지금 먹을 수 있는 걸로 골라 보자.",
                topic,
                "done",
                menu,
            )
        if yes:
            return respond(
                f"좋아, {detail} 쪽으로 마음이 가는구나. 맛있게 먹고 와!",
                topic,
                "done",
                detail,
            )
        if no:
            return respond(
                "다른 걸 골라 보자. 지금 떠오르는 음식은 있어?", topic, "choice"
            )

    # 사용자가 새 취미를 말했으면 그 취미로 주제를 바꾼다.
    hobby = re.search(r"게임|음악|영화|독서|그림|운동|축구|요리", text)
    if hobby and (topic != "hobby" or stage == "name"):
        subject = hobby[0]
        question = {
            "게임": "어떤 게임을 주로 해?",
            "음악": "요즘 자주 듣는 노래가 있어?",
            "영화": "최근에 본 영화 중 기억에 남는 게 있어?",
        }.get(subject, f"{subject}에서 어떤 점이 제일 좋아?")
        return respond(question, "hobby", "detail", subject)
    if topic == "hobby" and stage in ("name", "detail") and (yes or no):
        if no:
            return respond(
                "아직 떠오르는 게 없구나. 그럼 오늘 있었던 일 중 하나를 이야기해 볼까?",
                "day",
                "detail",
            )
        return respond("어떤 걸 좋아하는지 이름을 하나 알려줄래?", topic, stage, detail)
    if topic == "hobby":
        if stage == "name":
            return respond(
                f"“{short(text)}”에 관심이 있구나. 어떤 점이 재미있어?",
                topic,
                "experience",
                text,
            )
        if stage == "detail":
            return respond(
                f"“{short(text)}” 좋아하는구나. {detail} 하면서 가장 재미있었던 순간이 있어?",
                topic,
                "experience",
                text,
            )
        if stage == "experience":
            return respond(
                f"“{short(text)}” 같은 순간이 기억에 남는구나. 그걸 혼자 즐기는 편이야, 누군가와 같이 하는 편이야?",
                topic,
                "choice",
                detail,
            )
        if stage == "choice":
            return respond(
                (
                    "같이하는 사람과 분위기도 재미에 영향을 주겠네."
                    if re.search(r"같이|친구|함께", text)
                    else "네가 편하게 즐기는 방식이 있는 거구나."
                ),
                topic,
                "done",
                detail,
            )

    if not negated and re.search(r"힘들|피곤|지쳤|속상|서운|슬퍼|외로|우울", text):
        if topic == "feeling" and stage == "detail":
            return respond(
                "그 마음이 아직 남아 있구나. 지금은 쉬고 싶은지, 이야기하고 싶은지 네가 편한 쪽으로 해도 돼.",
                topic,
                "support",
                detail,
            )
        return respond(
            "마음이 편하지만은 않구나. 무슨 일이 있었는지 조금 더 이야기해 줄래?",
            "feeling",
            "detail",
            text,
        )
    if topic == "feeling":
        if re.search(r"그냥.*(?:들어|듣)|이야기|말하고", text):
            return respond(
                "응, 들어 줄게. 중간에 결론 내리지 않고 들을 테니 편하게 말해 줘.",
                topic,
                "listen",
                detail,
            )
        return respond(
            f"“{short(text)}”라는 일이 있었구나. 그중 어떤 부분이 가장 마음에 걸려?",
            topic,
            "listen",
            text,
        )
    if topic == "day":
        if stage == "day":
            if re.search(r"좋|재밌|행복|신나", text) and not negated:
                return respond(
                    "좋은 하루였구나! 어떤 일이 제일 좋았어?", topic, "detail", text
                )
            return respond(
                f"“{short(text)}” 그런 하루였구나. 오늘 기억에 남는 일이 있어?",
                topic,
                "detail",
                text,
            )
        if stage == "detail":
            return respond(
                f"“{short(text)}”라는 일이 있었구나. 그때는 어떤 기분이었어?",
                topic,
                "feeling",
                text,
            )
        if stage == "feeling":
            return respond(
                f"그 일이 있고 나서 “{short(text)}”라는 마음이 들었구나. 지금도 그 기분이 남아 있어?",
                topic,
                "done",
                detail,
            )
        if yes:
            return respond(
                "아직 그 여운이 남아 있구나. 조금 더 이야기하고 싶은 부분이 있어?",
                topic,
                "detail",
                detail,
            )
        if no:
            return respond(
                "지금은 마음이 달라졌구나. 그럼 지금은 어떤 얘기를 하고 싶어?"
            )
    if local_result["source"] != "unknown":
        return {**local_result, "dialogue": state}
    if yes or no:
        return respond(
            "어떤 부분에 대한 답인지 한마디만 더 붙여 줄래? 내가 잘 따라가고 싶어.",
            topic,
            stage,
            detail,
        )
    if "?" in text or re.search(r"왜|어떻게|알려\s*줘", text):
        return respond(
            "그 질문에는 아직 확실한 답을 못 하겠어. 어떤 상황에서 궁금해졌는지 말해 줄래?",
            "free",
            "detail",
            text,
        )
    if topic == "free" and stage == "detail":
        return respond(
            f"아까 한 이야기에서 이어지는 말이구나. “{short(text)}”라는 부분은 너한테 어떤 느낌이야?",
            "free",
            "feeling",
            text,
        )
    return respond(
        f"“{short(text)}”라고 했구나. 그 얘기를 조금 더 듣고 싶어. 어떤 일이 있었어?",
        "free",
        "detail",
        text,
    )
