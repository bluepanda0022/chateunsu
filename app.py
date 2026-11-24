import json
import os
import sys
import uuid
from collections import OrderedDict
from threading import Lock
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chatbot import ConversationBook
from config import load_env_file
from dictionary import Dictionaries
from conversation import ConversationEngine
from korean_tokenizer import tokenize
from local_model import LocalModel, ModelError

ROOT = Path(__file__).parent
MAX_REQUEST_BYTES = 100_000


def chat(messages, book, dictionaries=None, model=None):
    result = book.reply(messages[-1]["content"])
    if model and result["source"] != "learned":
        return model.complete(messages)
    if dictionaries is not None:
        return ConversationEngine(dictionaries).reply(messages, result)
    return result


def validate_messages(messages):
    """사용자·어시스턴트 메시지 목록의 API 입력 형식을 검사한다."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("올바른 대화가 필요합니다.")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        raise ValueError("마지막 메시지는 사용자 메시지여야 합니다.")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("올바른 대화가 필요합니다.")


class ChatServer(ThreadingHTTPServer):
    """대화 저장소를 소유하는 로컬 HTTP 서버."""

    def __init__(self, address, book: ConversationBook, dictionaries=None, model=None):
        self.book = book
        self.dictionaries = dictionaries if dictionaries is not None else Dictionaries()
        self.model = (
            (
                LocalModel(dictionaries=self.dictionaries)
                if os.getenv("CHAT_ENGINE", "ollama") != "rules"
                else None
            )
            if model is None
            else model
        )
        self.traces = OrderedDict()
        self.trace_lock = Lock()
        super().__init__(address, Handler)

    def remember_trace(self, result):
        trace = result.pop("_token_trace", None)
        if trace is not None:
            trace_id = uuid.uuid4().hex
            with self.trace_lock:
                self.traces[trace_id] = trace
                while len(self.traces) > 12:
                    self.traces.popitem(last=False)
            result["token_trace_id"] = trace_id
        return result


class Handler(SimpleHTTPRequestHandler):
    server: ChatServer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "static"), **kwargs)

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size < 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("입력은 100KB 이하로 보내 주세요.")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("JSON 객체가 필요합니다.")
            if self.path == "/api/tokenize":
                if not isinstance(data.get("text"), str):
                    raise ValueError("텍스트를 입력해 주세요.")
                mode = data.get("mode", "morpheme")
                if mode in ("inference-input", "inference-output"):
                    with self.server.trace_lock:
                        trace = self.server.traces.get(data.get("trace_id"))
                    if trace is None:
                        self.reply(
                            404,
                            {
                                "error": "추론 기록이 없어요. 서버 재시작 또는 기록 만료 후에는 새 답변을 생성해 주세요."
                            },
                        )
                        return
                    index = int(data.get("round_index", len(trace["rounds"]) - 1))
                    if not 0 <= index < len(trace["rounds"]):
                        raise ValueError("잘못된 추론 차수")
                    round = trace["rounds"][index]
                    part = (
                        "input_tokens" if mode == "inference-input" else "output_tokens"
                    )
                    text = (
                        round["prompt"]
                        if mode == "inference-input"
                        else round["response"]
                    )
                    result = {
                        "tokens": round[part],
                        "source_text": text,
                        "characters": len(text),
                        "bytes": len(text.encode()),
                        "mode": mode,
                        "token_origin": (
                            "backend-matched-input"
                            if mode == "inference-input"
                            and round["input_count_matches"]
                            and round["context_prefix_matches"]
                            else (
                                "rendered-input"
                                if mode == "inference-input"
                                else "reencoded-output"
                            )
                        ),
                        "method": f'{trace["model"]} · 실제 추론 {index + 1}차',
                        "model": trace["model"],
                        "round_index": index,
                        "rounds": len(trace["rounds"]),
                        "round_kinds": [
                            r.get("kind", "answer") for r in trace["rounds"]
                        ],
                        "verified": round["vocabulary_verified"],
                        "input_count_matches": round["input_count_matches"],
                        "context_prefix_matches": round["context_prefix_matches"],
                        "reported_input_tokens": round["reported_input_tokens"],
                        "reported_output_tokens": round["reported_output_tokens"],
                        "dictionary_lookups": trace["dictionary_lookups"],
                    }
                else:
                    if mode == "model" and not self.server.model:
                        raise ModelError(
                            "로컬 모델이 비활성화되어 있어요. CHAT_ENGINE=ollama로 실행해 주세요."
                        )
                    if mode == "model":
                        self.server.model.verify_tokenizer()
                    result = tokenize(
                        data["text"],
                        mode,
                        self.server.model.model if self.server.model else "qwen3",
                    )
                    if mode == "model" and self.server.model:
                        result["method"] = (
                            self.server.model.model + " · 검증된 BPE (입력 본문)"
                        )
                        result["verified"] = True
            elif self.path in ("/api/chat", "/api/chat/stream"):
                messages = data.get("messages")
                validate_messages(messages)
                if self.path.endswith("/stream"):
                    self.stream_chat(messages)
                    return
                result = chat(
                    messages,
                    self.server.book,
                    self.server.dictionaries,
                    self.server.model,
                )
            elif self.path == "/api/teach":
                result = self.server.book.teach(
                    data.get("question"), data.get("answer")
                )
            else:
                self.send_error(404)
                return
            self.reply(200, self.server.remember_trace(result))
        except ModelError as error:
            self.reply(503, {"error": str(error)})
        except (ImportError, RuntimeError) as error:
            self.reply(
                503,
                {
                    "error": "분석 도구를 준비하지 못했어요. .venv 환경에서 requirements.txt를 설치해 주세요."
                },
            )
        except (ValueError, TypeError, KeyError):
            self.reply(400, {"error": "입력 형식을 확인해 주세요."})
        except Exception:
            self.reply(
                502,
                {"error": "처리 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요."},
            )

    def stream_chat(self, messages):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event):
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        try:
            local = self.server.book.reply(messages[-1]["content"])
            if self.server.model and local["source"] != "learned":
                send({"type": "status", "content": "은수가 답변을 만들고 있어요…"})
                for event in self.server.model.stream(messages):
                    if event.get("type") == "done":
                        self.server.remember_trace(event["response"])
                    send(event)
            else:
                result = chat(
                    messages, self.server.book, self.server.dictionaries, False
                )
                send({"type": "done", "response": result})
        except ModelError as error:
            send({"type": "error", "error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            send(
                {
                    "type": "error",
                    "error": "답변 생성 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.",
                }
            )

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    venv_python = ROOT / ".venv" / "bin" / "python"
    if sys.prefix != str(ROOT / ".venv") and venv_python.is_file():
        os.execv(str(venv_python), [str(venv_python), str(ROOT / "app.py")])
    load_env_file(ROOT / ".env")
    from runtime import ensure_local_runtime

    ensure_local_runtime()
    with ChatServer(("127.0.0.1", 8000), ConversationBook()) as server:
        print("은수 실행: http://localhost:8000", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n은수를 종료합니다.")
