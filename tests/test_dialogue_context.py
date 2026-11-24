import unittest
from local_model import render_prompt
from word_selection import candidates_for


class DialogueContextTests(unittest.TestCase):
    def test_ordinary_game_and_house_messages_do_not_trigger_dictionary(self):
        for text in [
            "오늘 친구랑 마인크래프트 했어.",
            "같이 집 지었어.",
            "내가 무슨 게임 했다고 했지?",
        ]:
            self.assertEqual(candidates_for([{"role": "user", "content": text}]), [])

    def test_unfamiliar_adjective_and_explicit_common_word_can_be_searched(self):
        self.assertIn(
            "곰살궂다",
            candidates_for([{"role": "user", "content": "친구가 나보고 곰살궂대."}]),
        )
        self.assertIn(
            "집", candidates_for([{"role": "user", "content": "집 뜻 알려줘"}])
        )

    def test_grounding_notes_use_user_words_not_assistant_inventions(self):
        messages = [
            {"role": "user", "content": "친구랑 마인크래프트 했어."},
            {"role": "assistant", "content": "좀비 모드를 했나 보네!"},
            {"role": "user", "content": "내가 무슨 게임 했다고 했지?"},
        ]
        prompt = render_prompt(messages)
        system = prompt.split("<|im_start|>user")[0]
        self.assertIn("친구랑 마인크래프트 했어.", system)
        self.assertNotIn("좀비 모드", system)
        self.assertIn("대화 기억을 확인", system)

    def test_dictionary_is_reference_data_not_a_new_user_turn(self):
        prompt = render_prompt(
            [
                {"role": "user", "content": "그 말이 칭찬일까?"},
                {"role": "tool", "content": "검증된 뜻풀이"},
            ]
        )
        self.assertEqual(prompt.count("<|im_start|>user"), 1)
        self.assertIn("사전 참고 자료", prompt.split("<|im_start|>user")[0])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(prompt.count("<|im_start|>system"), 1)


if __name__ == "__main__":
    unittest.main()
