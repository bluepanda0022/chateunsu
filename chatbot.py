"""외부 서비스 없이 질문·답변을 검색하고, 사용자가 가르친 대화를 저장한다."""

import random
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path

DATABASE = Path(__file__).parent / "data" / "eunsu.sqlite3"
MAX_TEXT_LENGTH = 1000

# 같은 뜻의 질문들을 묶되, 짧은 단어의 부분 일치로 오답을 만들지 않는다.
SEEDS = [
    (
        ["안녕", "안녕하세요", "하이", "반가워", "안뇽"],
        ["안녕! 나는 은수야. 오늘 하루는 어땠어?", "어서 와! 무슨 얘기부터 해 볼까?"],
    ),
    (
        ["이름이 뭐야", "너 누구야", "자기소개 해줘", "이름 알려줘"],
        ["나는 은수야! 네가 가르쳐 준 말을 기억하면서 함께 수다 떠는 친구야."],
    ),
    (
        ["뭐해", "뭐 하고 있어", "머해"],
        ["너랑 이야기할 준비 중이었지. 너는 뭐 하고 있었어?"],
    ),
    (
        ["심심해", "심심하다", "놀아줘", "나 심심해"],
        [
            "그럼 우리 같이 놀자! 요즘 제일 빠져 있는 게 뭐야?",
            "내가 질문 하나 할게. 하루 동안 어디든 갈 수 있다면 어디로 가고 싶어?",
        ],
    ),
    (
        ["배고파", "배고프다", "나 배고파"],
        ["맛있는 거 먹자! 지금 밥이 당겨, 간식이 당겨?"],
    ),
    (
        ["뭐 먹을까", "메뉴 추천", "점심 뭐 먹지"],
        ["따뜻한 김치볶음밥 어때? 계란 프라이 하나 올리면 좋겠다."],
    ),
    (
        ["고마워", "감사해", "감사합니다", "고맙다"],
        ["천만에! 이렇게 말해 주니까 나도 기분 좋다.", "언제든지! 또 이야기해 줘."],
    ),
    (
        ["잘 자", "잘자", "좋은 밤", "나 잘게"],
        ["오늘도 수고했어. 푹 자고 내일 또 만나!"],
    ),
    (
        ["잘 가", "잘가", "바이", "다음에 봐"],
        ["다음에 또 와! 오늘 이야기해서 즐거웠어."],
    ),
    (["좋아해", "사랑해"], ["따뜻한 말 고마워! 나도 너랑 이야기하는 시간이 좋아."]),
    (
        ["힘들어", "오늘 너무 힘들어", "지쳤어"],
        ["오늘 많이 지쳤구나. 무슨 일이 있었는지 편하게 이야기해도 돼."],
    ),
    (
        ["슬퍼", "우울해", "속상해"],
        ["마음이 무거운 날이구나. 어떤 일이 있었어? 내가 들어 줄게."],
    ),
    (
        ["기분 좋아", "신나", "행복해"],
        ["좋다! 무슨 좋은 일이 있었어? 나한테도 들려줘."],
    ),
    (["ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "하하"], ["ㅎㅎ 같이 웃으니까 좋네!"]),
    (["응", "웅", "네", "그래"], ["좋아! 조금 더 이야기해 줘."]),
    (["아니", "아니야", "싫어"], ["그렇구나. 네 생각은 어떤지 알려줘."]),
    (["미안해", "미안", "죄송해요"], ["괜찮아. 우리 편하게 이야기하자."]),
    (
        ["날씨 어때", "오늘 날씨 알려줘"],
        ["나는 실시간 날씨를 볼 수는 없어. 네가 있는 곳은 맑아, 흐려?"],
    ),
    (
        ["토큰이 뭐야", "토크나이저가 뭐야", "한국어 토크나이저와 토큰은 무엇인가요"],
        [
            "토큰은 문장을 나눈 작은 단위야. 왼쪽 토크나이저 실험실에서 어절과 공백이 어떻게 나뉘는지 확인해 봐!"
        ],
    ),
    (
        ["어떻게 가르쳐", "사용법", "도움말"],
        [
            "답변 아래의 “답변 가르치기”를 누르고, 내가 어떻게 답하면 좋을지 적어 줘. 다음에 비슷한 말을 들으면 배운 답변을 찾아볼게."
        ],
    ),
]


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"[^\w]", "", text).replace("_", "")


class ConversationBook:
    def __init__(self, path=DATABASE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS replies (
                question TEXT NOT NULL, normalized TEXT NOT NULL,
                answer TEXT NOT NULL, UNIQUE(normalized, answer)
            )"""
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def teach(self, question, answer):
        for value in (question, answer):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > MAX_TEXT_LENGTH
            ):
                raise ValueError("질문과 답변은 1~1000자로 입력해 주세요.")
        key = normalize(question)
        if not key:
            raise ValueError("질문에는 글자나 숫자가 필요합니다.")
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO replies VALUES (?, ?, ?)",
                (question.strip(), key, answer.strip()),
            )
        return {"saved": bool(cursor.rowcount), "mode": "local"}

    def reply(self, question):
        key = normalize(question)
        with self.connect() as connection:
            learned = connection.execute(
                "SELECT normalized, answer FROM replies"
            ).fetchall()
        # 직접 가르친 답변이 기본 대화보다 우선한다.
        exact = [answer for candidate, answer in learned if candidate == key]
        if exact:
            return self.result(random.choice(exact), "learned")
        for questions, answers in SEEDS:
            if key in [normalize(q) for q in questions]:
                return self.result(random.choice(answers), "basic")

        candidates = [(candidate, answer, "learned") for candidate, answer in learned]
        candidates += [
            (normalize(q), answer, "basic")
            for questions, answers in SEEDS
            for q in questions
            for answer in answers
        ]
        best_score, best = 0, None
        for candidate, answer, source in candidates:
            # 짧은 질문은 의미가 쉽게 달라져 정확히 일치할 때만 응답한다.
            if min(len(key), len(candidate)) < 4:
                continue
            score = SequenceMatcher(None, key, candidate, autojunk=False).ratio()
            if score > best_score:
                best_score, best = score, (answer, source)
        if best and best_score >= 0.82:
            return self.result(*best)
        return self.result(
            "아직 그 말에는 어떻게 답하면 좋을지 모르겠어. 아래 “답변 가르치기”로 알려줄래?",
            "unknown",
        )

    @staticmethod
    def result(content, source):
        return {"content": content, "mode": "local", "source": source}
