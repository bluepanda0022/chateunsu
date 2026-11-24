import unittest
from tone import normalize_tone, polite_requested


class ToneTests(unittest.TestCase):
    def test_remove_honorifics_in_casual_reply(self):
        self.assertEqual(normalize_tone("뭐 하셨어?"), "뭐 했어?")
        self.assertEqual(normalize_tone("내일 발표예요."), "내일 발표야.")
        self.assertEqual(normalize_tone("어떤 게임을 하셨나요?"), "어떤 게임을 했어?")

    def test_consistent_polite_reply(self):
        self.assertEqual(normalize_tone("재미있었겠어.", True), "재미있었겠어요.")
        self.assertEqual(
            normalize_tone("오늘 친구랑 게임하셨네?", True), "오늘 친구랑 게임하셨네요?"
        )

    def test_repeated_question_only_reply_is_shortened(self):
        self.assertEqual(
            normalize_tone("그런 거 좋아해? 어떤 게임이야?"), "어떤 게임이야?"
        )

    def test_code_and_normal_sentences_are_preserved(self):
        text = '이렇게 해 봐.\n```python\nprint("안녕하세요")\n```'
        self.assertEqual(normalize_tone(text), text)
        self.assertEqual(normalize_tone("내일이야."), "내일이야.")

    def test_explicit_preference_and_changes(self):
        self.assertFalse(polite_requested([]))
        self.assertTrue(
            polite_requested([{"role": "user", "content": "존댓말로 해줘"}])
        )
        self.assertFalse(
            polite_requested(
                [
                    {"role": "user", "content": "존댓말로 해줘"},
                    {"role": "user", "content": "이제 반말로 해줘"},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
