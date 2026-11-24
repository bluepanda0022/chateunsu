import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

from app import ChatServer, chat, tokenize, validate_messages
from chatbot import ConversationBook
from local_model import ModelError


class TokenTests(unittest.TestCase):
    def test_source_offsets_preserve_original_text(self):
        for text in ["", "안녕하세요, 은수! 👋\nHello 123", "가_나\tabc", "한글"]:
            result = tokenize(text)
            self.assertEqual(result["source_text"], text)
            self.assertEqual(result["bytes"], len(text.encode()))
            for token in result["tokens"]:
                self.assertEqual(token["surface"], text[token["start"] : token["end"]])
                self.assertNotIn("id", token)

    def test_korean_morphemes_are_not_split_only_by_spaces(self):
        result = tokenize("먹었어요")
        self.assertEqual([t["text"] for t in result["tokens"]], ["먹", "었", "어요"])
        self.assertEqual([t["pos"] for t in result["tokens"]], ["VV", "EP", "EF"])
        self.assertGreater(len(tokenize("띄어쓰기없이도분석합니다")["tokens"]), 3)

    def test_real_qwen_token_ids_round_trip(self):
        from korean_tokenizer import model_tokenizer

        text = "한국어를 배워요! 👋"
        result = tokenize(text, "model")
        self.assertEqual(
            model_tokenizer().decode([t["id"] for t in result["tokens"]]), text
        )
        self.assertEqual([t["id"] for t in tokenize("나", "model")["tokens"]], [60315])

    def test_korean_byte_fragments_are_real_tokens_without_duplicate_bytes(self):
        text = "안녕하세요, 은수와 함께 한국어를 탐험해요!"
        result = tokenize(text, "model")
        self.assertEqual(result["token_count"], 18)
        self.assertEqual(result["fragment_tokens"], 4)
        self.assertEqual(
            b"".join(bytes.fromhex(t["byte_hex"]) for t in result["tokens"]),
            text.encode("utf-8"),
        )
        self.assertEqual(sum(t["bytes"] for t in result["tokens"]), len(text.encode()))
        self.assertEqual(len(tokenize(text, "morpheme")["tokens"]), 13)

    def test_literal_replacement_character_and_special_tokens_remain_visible(self):
        from model_trace import input_tokens

        text = "� 한글 👋\n<|im_start|>"
        tokens = input_tokens(text)
        special = [t for t in tokens if t["special"]]
        self.assertEqual([t["text"] for t in special], ["<|im_start|>"])
        self.assertTrue(
            any(t["text"] == "�" and not t["byte_fragment"] for t in tokens)
        )

    def test_tokenizer_limits(self):
        for text, mode in [("test", "invalid"), ("가" * 10001, "morpheme")]:
            with self.assertRaises(ValueError):
                tokenize(text, mode)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.sqlite3"
        self.book = ConversationBook(self.path)

    def test_basic_chat_without_external_model(self):
        result = chat([{"role": "user", "content": "안녕하세요!"}], self.book)
        self.assertEqual(result["mode"], "local")
        self.assertEqual(result["source"], "basic")

    def test_unknown_and_short_false_match(self):
        for question in ["양자컴퓨터의 오류정정에 대해 알려줘", "안", "안녕하지마"]:
            self.assertEqual(self.book.reply(question)["source"], "unknown")

    def test_teach_persist_normalize_and_deduplicate(self):
        self.assertTrue(
            self.book.teach("오늘 취미 시간", "나는 그림 그리기가 좋아!")["saved"]
        )
        self.assertFalse(
            self.book.teach("오늘 취미 시간!", "나는 그림 그리기가 좋아!")["saved"]
        )
        restarted = ConversationBook(self.path)
        result = restarted.reply("오늘취미시간?")
        self.assertEqual(result["content"], "나는 그림 그리기가 좋아!")
        self.assertEqual(result["source"], "learned")

    def test_learned_reply_overrides_seed(self):
        self.book.teach("안녕", "또 만났네!")
        self.assertEqual(self.book.reply("안녕")["content"], "또 만났네!")

    def test_fuzzy_long_question(self):
        self.book.teach("오늘 학교에서 무슨 일이 있었어", "친구랑 같이 점심 먹었어!")
        self.assertEqual(
            self.book.reply("오늘 학교에서 무슨 일이 있었니")["source"], "learned"
        )

    def test_invalid_teaching(self):
        for question, answer in [
            ("", "답"),
            ("!!!", "답"),
            ("질문", ""),
            ("질문", "a" * 1001),
            (None, "답"),
        ]:
            with self.assertRaises(ValueError):
                self.book.teach(question, answer)

    def test_invalid_messages(self):
        for messages in [[], [None], [{"role": "assistant", "content": "답"}]]:
            with self.assertRaises(ValueError):
                validate_messages(messages)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.book = ConversationBook(Path(self.directory.name) / "test.sqlite3")
        self.server = ChatServer(("127.0.0.1", 0), self.book, model=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def post(self, path, data):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=5)

    def test_teach_then_chat_over_http(self):
        with self.post(
            "/api/teach", {"question": "우리만의 인사", "answer": "은수 왔어!"}
        ) as response:
            self.assertTrue(json.load(response)["saved"])
        with self.post(
            "/api/chat", {"messages": [{"role": "user", "content": "우리만의 인사"}]}
        ) as response:
            result = json.load(response)
            self.assertEqual(result["content"], "은수 왔어!")
            self.assertEqual(result["mode"], "local")

    def test_invalid_input_returns_json(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/chat", {})
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("error", json.load(response))

    def test_daily_dialogue_state_round_trip_without_dictionary(self):
        self.server.dictionaries = Mock()
        self.server.dictionaries.lookup.side_effect = AssertionError(
            "Unexpected dictionary call"
        )
        messages = []
        for prompt, expected in [
            ("배고파", "choice"),
            ("밥", "menu"),
            ("매콤한 거", "done"),
            ("응", "done"),
        ]:
            messages.append({"role": "user", "content": prompt})
            with self.post("/api/chat", {"messages": messages}) as response:
                answer = json.load(response)
            self.assertEqual(answer["mode"], "local")
            self.assertEqual(answer["dialogue"]["stage"], expected)
            self.assertFalse(answer["sources"])
            messages.append({"role": "assistant", **answer})
        self.assertIn("김치볶음밥", answer["content"])
        self.server.dictionaries.lookup.assert_not_called()

    def test_model_stream_over_http(self):
        self.server.model = Mock()
        self.server.model.stream.return_value = iter(
            [
                {"type": "delta", "content": "안녕"},
                {"type": "delta", "content": "!"},
                {"type": "done", "response": {"content": "안녕!", "mode": "model"}},
            ]
        )
        with self.post(
            "/api/chat/stream", {"messages": [{"role": "user", "content": "안녕"}]}
        ) as response:
            self.assertIn("application/x-ndjson", response.headers["Content-Type"])
            events = [json.loads(line) for line in response]
        self.assertEqual(events[-1]["response"]["content"], "안녕!")
        self.assertEqual(
            "".join(e.get("content", "") for e in events if e["type"] == "delta"),
            "안녕!",
        )

    def test_model_failure_is_an_error_event(self):
        self.server.model = Mock()
        self.server.model.stream.side_effect = ModelError("모델 연결 실패")
        with self.post(
            "/api/chat/stream", {"messages": [{"role": "user", "content": "안녕"}]}
        ) as response:
            events = [json.loads(line) for line in response]
        self.assertEqual(events[-1], {"type": "error", "error": "모델 연결 실패"})
        self.assertFalse(any(event["type"] == "done" for event in events))

    def test_trace_can_be_opened_in_tokenizer_lab(self):
        trace = {
            "model": "test-model",
            "dictionary_lookups": [],
            "rounds": [
                {
                    "kind": "answer",
                    "prompt": "전체 입력",
                    "response": "안녕",
                    "input_tokens": [{"id": 1, "text": "전체"}],
                    "output_tokens": [{"id": 2, "text": "안녕"}],
                    "vocabulary_verified": True,
                    "input_count_matches": True,
                    "context_prefix_matches": True,
                    "reported_input_tokens": 1,
                    "reported_output_tokens": 2,
                }
            ],
        }
        result = self.server.remember_trace({"content": "안녕", "_token_trace": trace})
        self.assertNotIn("_token_trace", result)
        with self.post(
            "/api/tokenize",
            {
                "text": "",
                "mode": "inference-input",
                "trace_id": result["token_trace_id"],
            },
        ) as response:
            data = json.load(response)
        self.assertEqual(data["source_text"], "전체 입력")
        self.assertEqual(data["tokens"][0]["id"], 1)
        self.assertTrue(data["verified"])
        self.assertEqual(data["round_kinds"], ["answer"])

    def test_trace_expiration_is_explicit(self):
        first = self.server.remember_trace({"_token_trace": {"test": 0}})[
            "token_trace_id"
        ]
        for i in range(12):
            self.server.remember_trace({"_token_trace": {"test": i}})
        self.assertEqual(len(self.server.traces), 12)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(
                "/api/tokenize",
                {"text": "", "mode": "inference-input", "trace_id": first},
            )
        with caught.exception as response:
            self.assertEqual(response.code, 404)
            self.assertIn("추론 기록", json.load(response)["error"])


if __name__ == "__main__":
    unittest.main()
