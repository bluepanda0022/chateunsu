import tempfile
import unittest
from pathlib import Path

from app import chat
from chatbot import ConversationBook
from conversation import conversational_definition, keywords


def entry(word, definition):
    return {
        "dictionary": "표준국어대사전",
        "word": word,
        "definition": definition,
        "pos": "명사",
        "url": "https://stdict.korean.go.kr/search/searchView.do?word_no=1",
    }


class FakeDictionaries:
    enabled = True

    def __init__(self):
        self.calls = []
        self.entries = {
            "사랑": [entry("사랑", "어떤 사람이나 존재를 아끼는 마음.")],
            "우정": [entry("우정", "친구 사이의 정.")],
            "나무": [entry("나무", "줄기나 가지가 목질로 된 여러해살이 식물.")],
            "은행": [
                entry("은행", "은행나무의 열매."),
                entry("은행", "돈을 맡아 주는 기관."),
            ],
            "돈": [entry("돈", "물건의 값을 나타내는 화폐.")],
            "다투다": [entry("다투다", "서로 따지며 싸우다.")],
            "고양이": [entry("고양이", "고양잇과의 포유류.")],
        }

    def lookup(self, query, selected=None):
        self.calls.append(query)
        return {
            "content": "검색 결과",
            "mode": "dictionary",
            "source": "dictionary",
            "sources": self.entries.get(query, []),
        }


class ConversationEngineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.book = ConversationBook(Path(directory.name) / "test.sqlite3")
        self.dictionaries = FakeDictionaries()
        self.messages = []

    def ask(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        answer = chat(self.messages, self.book, self.dictionaries)
        self.messages.append({"role": "assistant", **answer})
        return answer

    def test_comparison_keeps_love_word_intact(self):
        result = self.ask("사랑과 우정 뜻 차이 알려줘")
        self.assertEqual(result["topics"], ["사랑", "우정"])
        self.assertNotIn("사", self.dictionaries.calls)
        self.assertIn("아끼는 마음이야", result["content"])
        self.assertIn("친구 사이의 정이야", result["content"])
        self.assertEqual(result["style"], "conversation")
        self.assertEqual(len(result["sources"]), 2)

    def test_followup_uses_previous_topics(self):
        self.ask("사랑과 우정 뜻 차이 알려줘")
        result = self.ask("예를 들어줘")
        self.assertEqual(result["topics"], ["사랑", "우정"])
        self.assertIn("내가 만든 예", result["content"])
        self.assertIn("사전 용례를 인용한 건 아니야", result["content"])
        self.assertNotIn("들어줘", self.dictionaries.calls)

    def test_followup_without_context_asks_for_topic(self):
        result = self.ask("예를 들어줘")
        self.assertEqual(result["mode"], "local")
        self.assertEqual(self.dictionaries.calls, [])

    def test_new_question_changes_topic(self):
        self.ask("사랑 뜻 알려줘")
        result = self.ask("나무 뜻 알려줘")
        self.assertEqual(result["topics"], ["나무"])

    def test_homonym_sense_is_chosen_from_sentence(self):
        self.ask("은행에 돈을 맡기고 싶어")
        self.assertEqual(self.dictionaries.calls, [])
        result = self.ask("은행 뜻 알려줘")
        self.assertIn("돈을 맡아 주는 기관", result["content"])
        self.assertNotIn("은행나무의 열매", result["content"])
        followup = self.ask("더 알려줘")
        self.assertIn("돈을 맡아 주는 기관", followup["content"])

    def test_emotional_message_gets_conversation(self):
        result = self.ask("친구랑 싸워서 속상해")
        self.assertIn("어떤 일로", result["content"])
        self.assertEqual(result["dialogue"]["topic"], "conflict")
        self.assertFalse(result["sources"])
        self.assertEqual(self.dictionaries.calls, [])

    def test_missing_comparison_side_is_not_invented(self):
        result = self.ask("사랑과 미등록어 뜻 차이 알려줘")
        self.assertIn("차이를 단정할 수 없어", result["content"])
        self.assertEqual(len(result["sources"]), 1)

    def test_greetings_and_taught_reply_skip_dictionary(self):
        self.assertEqual(self.ask("안녕")["source"], "basic")
        self.book.teach("우리 약속", "금요일에 이야기하자")
        self.assertEqual(self.ask("우리 약속")["content"], "금요일에 이야기하자")
        self.assertEqual(self.dictionaries.calls, [])

    def test_explicit_search_keeps_raw_result(self):
        self.assertEqual(
            self.ask("표준국어대사전에서 나무 찾아줘")["content"], "검색 결과"
        )

    def test_cannot_invent_causes_from_dictionary(self):
        result = self.ask("왜 고양이는 그렇게 행동해?")
        self.assertIn("확실한 답을 못", result["content"])
        self.assertEqual(self.dictionaries.calls, [])

    def test_negated_emotion_does_not_assume_feeling(self):
        result = self.ask("요즘은 외롭지 않아")
        self.assertNotIn("외로운 마음이 들었구나", result["content"])

    def test_daily_conversation_never_looks_up_definitions(self):
        for prompt in ["안녕", "좋았어", "친구랑 놀았어", "재밌었어", "응"]:
            result = self.ask(prompt)
            self.assertNotIn("사전", result["content"])
            self.assertFalse(result.get("sources"))
        self.assertEqual(self.dictionaries.calls, [])
        self.assertEqual(result["dialogue"]["topic"], "day")

    def test_conflict_conversation_moves_forward(self):
        for prompt, stage in [
            ("친구랑 싸웠어", "reason"),
            ("내 말을 무시했어", "feeling"),
            ("너무 속상했어", "support"),
            ("그냥 들어줘", "listen"),
        ]:
            result = self.ask(prompt)
            self.assertEqual(result["dialogue"]["stage"], stage)
            self.assertNotIn("라는 뜻", result["content"])
        self.assertEqual(self.dictionaries.calls, [])
        self.assertIn("조언부터 하지 않고", result["content"])

    def test_food_choices_continue_instead_of_generic_yes_reply(self):
        self.ask("배고파")
        self.assertIn("매콤", self.ask("밥")["content"])
        self.assertIn("김치볶음밥", self.ask("매콤한 거")["content"])
        self.assertIn("김치볶음밥", self.ask("응")["content"])
        self.assertEqual(self.dictionaries.calls, [])

    def test_hobby_name_is_followed_up(self):
        self.ask("심심해")
        self.assertIn("어떤 게임", self.ask("게임")["content"])
        self.assertIn("마인크래프트", self.ask("마인크래프트")["content"])
        self.assertIn("기억에 남는", self.ask("친구랑 건축할 때")["content"])
        self.assertEqual(self.dictionaries.calls, [])

    def test_returning_to_chat_after_dictionary_does_not_repeat_definition(self):
        self.ask("사랑 뜻 알려줘")
        calls = list(self.dictionaries.calls)
        result = self.ask("친구랑 싸웠어")
        self.assertEqual(result["mode"], "local")
        self.assertFalse(result["sources"])
        self.assertEqual(self.dictionaries.calls, calls)

    def test_do_not_define_when_user_says_not_to(self):
        self.ask("사랑 뜻 알려줘")
        calls = list(self.dictionaries.calls)
        self.assertEqual(self.ask("뜻 말고 그냥 대화하자")["mode"], "local")
        self.assertEqual(self.dictionaries.calls, calls)

    def test_new_conversation_does_not_inherit_state(self):
        self.ask("친구랑 싸웠어")
        self.messages = []
        result = self.ask("안녕")
        self.assertEqual(result["dialogue"]["topic"], "day")

    def test_conservative_particle_handling(self):
        self.assertEqual(keywords("고양이 이야기"), ["고양이"])
        self.assertEqual(keywords("은행에 돈을 맡기고 싶어"), ["은행", "돈"])
        self.assertEqual(keywords("사랑과 우정은"), ["사랑", "우정"])

    def test_conversational_endings_keep_original_meaning(self):
        self.assertEqual(
            conversational_definition("친구 사이의 정."), "친구 사이의 정이야."
        )
        self.assertEqual(
            conversational_definition("마음이 답답하다."), "마음이 답답해."
        )


if __name__ == "__main__":
    unittest.main()
