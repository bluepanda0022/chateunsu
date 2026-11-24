import io
import json
import ssl
import tempfile
import threading
import urllib.request
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import Mock, patch

from app import ChatServer, chat
from chatbot import ConversationBook
from dictionary import (
    Dictionaries,
    PROVIDERS,
    dictionary_ssl_context,
    extract_query,
    parse_response,
)

# 공식 검색 응답 구조를 따른 합성 데이터. 실제 API 호출은 사용하지 않는다.
STANDARD_XML = """<xml version="2.0"><channel><total>1</total><item>
<word>테스트</word><pos>명사</pos><sense><definition>검증용 뜻풀이.</definition>
<link>https://stdict.korean.go.kr/search/searchView.do?word_no=1</link>
</sense></item></channel></xml>""".encode()
OPEN_XML = """<channel><total>1</total><item><word>테스트</word><sense>
<pos>명사</pos><definition>두 번째 검증용 뜻풀이.</definition>
<link>https://opendict.korean.go.kr/dictionary/view?sense_no=2</link>
</sense></item></channel>""".encode()
EMPTY_XML = b"<channel><total>0</total></channel>"


def successful_opener(request, timeout):
    if urllib.parse.urlsplit(request.full_url).hostname == "stdict.korean.go.kr":
        return io.BytesIO(STANDARD_XML)
    return io.BytesIO(OPEN_XML)


class DictionaryTests(unittest.TestCase):
    def test_tls_certificate_failure_has_specific_message(self):
        error = urllib.error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )
        client = Dictionaries({"stdict": "test-key"}, Mock(side_effect=error))
        result = client.lookup("나무", "stdict")
        self.assertIn("HTTPS 인증서 검증에 실패", result["content"])
        self.assertNotIn("test-key", result["content"])

    def test_macos_certificates_are_loaded_without_disabling_tls(self):
        dictionary_ssl_context.cache_clear()
        self.addCleanup(dictionary_ssl_context.cache_clear)
        context = ssl.create_default_context()
        with (
            patch("dictionary.sys.platform", "darwin"),
            patch("dictionary.ssl.create_default_context", return_value=context),
            patch(
                "dictionary.subprocess.run", return_value=Mock(returncode=0, stdout="")
            ) as run,
        ):
            self.assertIs(dictionary_ssl_context(), context)
            self.assertEqual(run.call_count, 2)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_extract_dictionary_intent(self):
        for prompt, expected in [
            ("사랑 뜻 알려줘", ("사랑", None)),
            ("정의 뜻", ("정의", None)),
            ("“사랑”의 뜻 알려줘", ("사랑", None)),
            ("나무가 뭐야?", ("나무", None)),
            ("바람이란?", ("바람", None)),
            ("사전 검색: 작은 집", ("작은 집", None)),
            ("표준국어대사전에서 사과 검색해줘", ("사과", "stdict")),
            ("우리말샘에서 바람 찾아줘", ("바람", "opendict")),
            ("안녕", None),
            ("너 누구야", None),
            ("사전 " + "가" * 81, None),
        ]:
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_query(prompt), expected)

    def test_both_official_shapes(self):
        standard = parse_response(STANDARD_XML, PROVIDERS[0])[0]
        opened = parse_response(OPEN_XML, PROVIDERS[1])[0]
        self.assertEqual(standard["pos"], "명사")
        self.assertEqual(opened["pos"], "명사")
        self.assertEqual(standard["dictionary"], "표준국어대사전")
        self.assertEqual(opened["dictionary"], "우리말샘")

    def test_search_both_providers_and_preserve_sources(self):
        client = Dictionaries(
            {"stdict": "secret-a", "opendict": "secret-b"}, successful_opener
        )
        result = client.lookup("테스트")
        self.assertEqual(len(result["sources"]), 2)
        self.assertIn("검증용 뜻풀이", result["content"])
        self.assertNotIn("secret-", json.dumps(result))

    def test_request_is_exact_utf8_encoded_search(self):
        calls = []

        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return io.BytesIO(EMPTY_XML)

        client = Dictionaries({"stdict": "test-key"}, opener)
        client.lookup("작은 집", "stdict")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0][0]).query)
        self.assertEqual(query["q"], ["작은 집"])
        self.assertEqual(query["advanced"], ["y"])
        self.assertEqual(query["method"], ["exact"])
        self.assertEqual(query["req_type"], ["xml"])
        self.assertEqual(calls[0][1], 6)

    def test_missing_keys_does_not_request_network(self):
        opener = Mock(side_effect=AssertionError("network should not be called"))
        result = Dictionaries({}, opener).lookup("단어")
        self.assertEqual(result["sources"], [])
        self.assertIn("STDICT_API_KEY", result["content"])
        self.assertIn("OPENDICT_API_KEY", result["content"])
        opener.assert_not_called()

    def test_api_auth_error_is_distinct_from_no_results(self):
        opener = lambda *args, **kwargs: io.BytesIO(
            b"<error><error_code>020</error_code><message>Unregistered key</message></error>"
        )
        result = Dictionaries({"stdict": "secret-key"}, opener).lookup("단어", "stdict")
        self.assertIn("인증 키가 유효하지", result["content"])
        self.assertNotIn("secret-key", result["content"])
        self.assertNotIn("일치하는 표제어", result["content"])

    def test_partial_failure_keeps_other_dictionary(self):
        def opener(request, timeout):
            if "stdict.korean.go.kr" in request.full_url:
                raise urllib.error.URLError("secret URL must not escape")
            return io.BytesIO(OPEN_XML)

        result = Dictionaries({"stdict": "a", "opendict": "b"}, opener).lookup("테스트")
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("표준국어대사전: 연결하지 못했어요", result["content"])
        self.assertNotIn("secret", result["content"])

    def test_empty_timeout_and_malformed_response(self):
        for body, expected in [
            (EMPTY_XML, "일치하는 표제어"),
            (b"<html>", "올바른 사전 응답"),
            (b"<html/>", "응답 형식"),
        ]:
            client = Dictionaries(
                {"stdict": "a"}, lambda *args, **kwargs: io.BytesIO(body)
            )
            self.assertIn(expected, client.lookup("단어", "stdict")["content"])
        client = Dictionaries({"stdict": "a"}, Mock(side_effect=TimeoutError()))
        self.assertIn("연결하지 못했어요", client.lookup("단어", "stdict")["content"])

    def test_unsafe_source_links_are_omitted(self):
        for url in [
            "javascript:alert(1)",
            "https://example.com/word",
            "https://stdict.korean.go.kr/api/search.do?key=secret",
        ]:
            body = STANDARD_XML.replace(
                b"https://stdict.korean.go.kr/search/searchView.do?word_no=1",
                url.encode(),
            )
            self.assertEqual(parse_response(body, PROVIDERS[0])[0]["url"], "")

    def test_chat_routing_and_local_learning_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            book = ConversationBook(Path(directory) / "test.sqlite3")
            client = Dictionaries({"stdict": "a", "opendict": "b"}, successful_opener)

            def ask(prompt):
                return chat([{"role": "user", "content": prompt}], book, client)

            self.assertEqual(ask("안녕")["source"], "basic")
            self.assertEqual(ask("테스트 뜻 알려줘")["source"], "dictionary")
            self.assertEqual(
                ask("우리말샘에서 테스트 찾아줘")["sources"][0]["dictionary"],
                "우리말샘",
            )
            self.assertEqual(ask("미등록단어")["mode"], "local")
            book.teach("미등록단어", "직접 배운 답변")
            self.assertEqual(ask("미등록단어")["content"], "직접 배운 답변")


class DictionaryHTTPTests(unittest.TestCase):
    def test_dictionary_sources_reach_chat_response(self):
        with tempfile.TemporaryDirectory() as directory:
            book = ConversationBook(Path(directory) / "test.sqlite3")
            client = Dictionaries({"stdict": "a", "opendict": "b"}, successful_opener)
            with ChatServer(("127.0.0.1", 0), book, client, model=False) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/chat",
                        data=json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "테스트 뜻 알려줘"}
                                ]
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        result = json.load(response)
                    self.assertEqual(result["mode"], "dictionary")
                    self.assertGreaterEqual(len(result["sources"]), 1)
                    self.assertTrue(
                        result["sources"][0]["url"].startswith(
                            "https://stdict.korean.go.kr/"
                        )
                    )
                finally:
                    server.shutdown()
                    thread.join()


if __name__ == "__main__":
    unittest.main()
