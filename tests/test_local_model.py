import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from app import chat
from chatbot import ConversationBook
from korean_tokenizer import model_tokenizer
from local_model import LocalModel, ModelError
from model_trace import verify_vocabulary


def response(*events):
    return io.BytesIO(
        b"".join(
            json.dumps(event, ensure_ascii=False).encode() + b"\n" for event in events
        )
    )


def client(opener, dictionaries=None):
    model = LocalModel(model="eunsu:qwen3-4b", opener=opener, dictionaries=dictionaries)
    # 모델 어휘 대조는 별도 테스트; 이 테스트들은 생성 프로토콜을 검증한다.
    model._verified = True
    return model


def generation(text, count=12):
    return response(
        {"response": text, "done": True, "prompt_eval_count": 42, "eval_count": count}
    )


class LocalModelTests(unittest.TestCase):
    def test_generated_text_is_not_reassembled_by_tone_filter(self):
        events = list(
            client(Mock(return_value=generation("내일 발표예요."))).stream(
                [{"role": "user", "content": "언제 발표야?"}]
            )
        )
        result = events[-1]["response"]
        self.assertEqual(result["content"], "내일 발표예요.")
        self.assertFalse(result["tone_adjusted"])
        self.assertFalse(any(event["type"] == "reset" for event in events))
        self.assertEqual(
            result["_token_trace"]["rounds"][-1]["response"], result["content"]
        )

    def test_history_is_sent_and_reply_is_generated_in_chunks(self):
        opener = Mock(
            return_value=response(
                {"response": "어제 발표가 ", "done": False},
                {
                    "response": "걱정됐구나.",
                    "done": True,
                    "prompt_eval_count": 42,
                    "eval_count": 8,
                },
            )
        )
        model = client(opener)
        messages = [
            {"role": "user", "content": "내일 발표야"},
            {"role": "assistant", "content": "어떤 점이 걱정돼?"},
            {"role": "user", "content": "사람들 앞에서 말하는 거"},
        ]
        events = list(model.stream(messages))
        self.assertEqual(events[-1]["response"]["content"], "어제 발표가 걱정됐구나.")
        payload = json.loads(opener.call_args.args[0].data)
        for message in messages:
            self.assertIn(message["content"], payload["prompt"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["template"], "{{ .Prompt }}")
        self.assertEqual(events[-1]["response"]["usage"]["output_tokens"], 8)

    def test_no_silent_template_fallback_when_model_is_down(self):
        with self.assertRaises(ModelError):
            client(Mock(side_effect=urllib.error.URLError("down"))).complete(
                [{"role": "user", "content": "안녕"}]
            )

    def test_missing_model_has_actionable_error(self):
        error = urllib.error.HTTPError("http://localhost", 404, "not found", {}, None)
        with self.assertRaisesRegex(ModelError, "ollama create"):
            client(Mock(side_effect=error)).complete(
                [{"role": "user", "content": "안녕"}]
            )

    def test_incomplete_stream_is_not_a_completed_answer(self):
        with self.assertRaises(ModelError):
            client(
                Mock(return_value=response({"response": "일부", "done": False}))
            ).complete([{"role": "user", "content": "안녕"}])

    def test_external_model_server_is_not_allowed(self):
        with self.assertRaises(ValueError):
            LocalModel(base_url="https://example.com")

    def test_history_window_and_prompt_budget(self):
        messages = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": "가" * 400,
                "sources": ["not sent"],
            }
            for i in range(40)
        ]
        model = LocalModel()
        history = model.history(messages)
        self.assertLessEqual(sum(len(m["content"]) for m in history), 3800)
        self.assertTrue(all(set(m) == {"role", "content"} for m in history))
        payload = model.payload(history)
        self.assertLessEqual(
            len(model_tokenizer(model.model).encode(payload["prompt"]).ids), 3000
        )

    def test_model_decides_to_search_without_user_asking_for_definition(self):
        dictionary = Mock()
        dictionary.lookup.return_value = {
            "content": "사전 응답",
            "sources": [
                {
                    "word": "도탑다",
                    "dictionary": "표준국어대사전",
                    "definition": "서로의 관계에 사랑이나 인정이 많고 깊다.",
                    "url": "https://stdict.korean.go.kr/word",
                }
            ],
        }
        opener = Mock(
            side_effect=[
                generation('{"words":["도탑다"]}'),
                generation("정이 많다는 좋은 말로 이해할 수 있어."),
            ]
        )
        result = client(opener, dictionary).complete(
            [
                {
                    "role": "user",
                    "content": "그 사람이 도탑다고 해서 기분 나빠할 말인지 모르겠어.",
                }
            ]
        )
        dictionary.lookup.assert_called_once_with("도탑다")
        second = json.loads(opener.call_args_list[1].args[0].data)
        self.assertIn("사랑이나 인정이 많고 깊다", second["prompt"])
        self.assertEqual(
            result["dictionary_lookups"], [{"word": "도탑다", "found": True}]
        )
        self.assertEqual(len(result["_token_trace"]["rounds"]), 2)
        self.assertEqual(
            result["_token_trace"]["rounds"][0]["kind"], "dictionary-decision"
        )

    def test_dictionary_request_does_not_search_request_words(self):
        payload = LocalModel().decision_payload(
            [{"role": "user", "content": "사랑 뜻 알려줘"}]
        )
        choices = payload["format"]["properties"]["words"]["items"]["enum"]
        self.assertIn("사랑", choices)
        self.assertNotIn("뜻", choices)

    def test_greeting_does_not_search(self):
        dictionary = Mock()
        result = client(
            Mock(return_value=generation("안녕!")),
            dictionary,
        ).complete([{"role": "user", "content": "안녕"}])
        dictionary.lookup.assert_not_called()
        self.assertEqual(result["content"], "안녕!")

    def test_invalid_and_duplicate_words_are_bounded(self):
        dictionary = Mock()
        dictionary.lookup.return_value = {"content": "없음", "sources": []}
        client(
            Mock(
                side_effect=[
                    generation('{"words":["낱말","낱말","추가"]}'),
                    generation("확인 못했어."),
                ]
            ),
            dictionary,
        ).complete([{"role": "user", "content": "낱말 얘기"}])
        dictionary.lookup.assert_called_once_with("낱말")
        dictionary.reset_mock()
        client(
            Mock(
                side_effect=[
                    generation('{"words":["https://example.com"]}'),
                    generation("안녕"),
                ]
            ),
            dictionary,
        ).complete([{"role": "user", "content": "주소"}])
        dictionary.lookup.assert_not_called()

    def test_vocabulary_mismatch_is_not_labelled_actual(self):
        with patch("model_trace.model_tokenizer") as tokenizer:
            tokenizer.return_value.get_vocab.return_value = {"a": 0, "b": 1}
            tokenizer.return_value.to_str.return_value = (
                '{"model":{"merges":[["a","b"]]}}'
            )
            self.assertTrue(
                verify_vocabulary(
                    {
                        "model_info": {
                            "tokenizer.ggml.tokens": ["a", "b"],
                            "tokenizer.ggml.merges": ["a b"],
                        }
                    }
                )
            )
            self.assertFalse(
                verify_vocabulary(
                    {
                        "model_info": {
                            "tokenizer.ggml.tokens": ["a", "wrong"],
                            "tokenizer.ggml.merges": ["a b"],
                        }
                    }
                )
            )
        with self.assertRaisesRegex(ModelError, "어휘"):
            LocalModel(
                opener=Mock(return_value=io.BytesIO(b'{"model_info":{}}'))
            ).verify_tokenizer()

    def test_both_ordinary_and_definition_requests_go_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            book = ConversationBook(Path(directory) / "book.sqlite3")
            model = Mock()
            model.complete.return_value = {
                "content": "자연스러운 답변",
                "mode": "model",
            }
            dictionary = Mock()
            for prompt in ["친구랑 싸웠어", "사과 뜻 알려줘"]:
                self.assertEqual(
                    chat(
                        [{"role": "user", "content": prompt}], book, dictionary, model
                    )["mode"],
                    "model",
                )
            self.assertEqual(model.complete.call_count, 2)
            dictionary.lookup.assert_not_called()
            book.teach("우리 약속", "내일 만나")
            self.assertEqual(
                chat(
                    [{"role": "user", "content": "우리 약속"}], book, dictionary, model
                )["content"],
                "내일 만나",
            )


if __name__ == "__main__":
    unittest.main()
