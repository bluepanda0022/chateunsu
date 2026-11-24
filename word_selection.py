"""일상 어휘를 사전 도구 후보에서 제외하고 활용형을 기본형으로 복원한다."""

from korean_tokenizer import analyzer, LOCK
from conversation import dictionary_requested

COMMON = set(
    "친구 사람 게임 집 오늘 내일 어제 일 말 기분 학교 회사 시간 공부 컴퓨터 영화 음악 노래 책 밥 음식 공원 운동 산책 대화 이야기 질문 답변 내용 이름 발표 걱정 기억 모드 재료 디자인 설계 건축 경우 생각 종류 과정 상황 준비 처음 재미 놀이 하나 나 너 안녕 의미 뜻 하다 가다 오다 놀다 짓다 만들다 좋다 좋아하다 먹다 자다 보다 듣다 읽다 쓰다 알다 모르다 말하다 되다 있다 없다 싶다 기억하다 걱정하다 긴장하다 즐기다 나다 준비하다 까먹다 모이다 배우다 같이 함께 그냥 조금 많이 계속 너무 매우 더 다시 아주 잘 어때 어떻다 얘기하다 아니다 필요하다 궁금하다 자세하다".split()
)


def candidates_for(messages):
    explicit = dictionary_requested(messages)
    with LOCK:
        tokens = analyzer().tokenize(messages[-1]["content"])
    candidates = []
    for token in tokens:
        if not token.tag.startswith(("NNG", "VV", "VA", "MAG", "XR")) and not (
            explicit and token.tag.startswith("NNP")
        ):
            continue
        word = token.form + "다" if token.tag.startswith(("VV", "VA")) else token.form
        if (not explicit and word in COMMON) or (len(word) < 2 and not explicit):
            continue
        if word not in candidates:
            candidates.append(word)
    meaningful = [
        word
        for word in candidates
        if word
        not in {"뜻", "의미", "정의", "알리다", "하다", "찾다", "말하다", "주다"}
    ]
    return (meaningful or candidates)[:30]
