"""국립국어원 사전 검색. 인증 키와 원격 오류 URL은 클라이언트에 노출하지 않는다."""

import os
import re
import ssl
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Provider:
    code: str
    name: str
    endpoint: str
    env: str
    host: str


PROVIDERS = (
    Provider(
        "stdict",
        "표준국어대사전",
        "https://stdict.korean.go.kr/api/search.do",
        "STDICT_API_KEY",
        "stdict.korean.go.kr",
    ),
    Provider(
        "opendict",
        "우리말샘",
        "https://opendict.korean.go.kr/api/search",
        "OPENDICT_API_KEY",
        "opendict.korean.go.kr",
    ),
)


@lru_cache(maxsize=1)
def dictionary_ssl_context():
    """Python 기본 CA와 macOS 시스템 인증서를 사용하며 TLS 검증은 유지한다."""
    context = ssl.create_default_context()
    if sys.platform == "darwin":
        for keychain in (
            "/System/Library/Keychains/SystemRootCertificates.keychain",
            "/Library/Keychains/System.keychain",
        ):
            try:
                result = subprocess.run(
                    ["/usr/bin/security", "find-certificate", "-a", "-p", keychain],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    context.load_verify_locations(cadata=result.stdout)
            except (OSError, subprocess.TimeoutExpired, ssl.SSLError):
                # 사용할 수 없는 키체인은 건너뛰되 기본 인증서 검증은 유지한다.
                continue
    return context


def open_dictionary(request, timeout):
    return urllib.request.urlopen(
        request, timeout=timeout, context=dictionary_ssl_context()
    )


def extract_query(prompt):
    """사전 검색 의도가 명확한 문장만 추출한다. 단순 수다는 로컬에 맡긴다."""
    text = unicodedata.normalize("NFC", prompt).strip().rstrip("?!？！.。 ")
    selected = None
    for provider in PROVIDERS:
        if text.startswith(provider.name):
            selected = provider.code
            text = text[len(provider.name) :].strip()
            text = re.sub(r"^(?:에서|으로)\s*", "", text)
            break
    text = re.sub(r"^(?:사전\s*검색|뜻풀이|뜻)\s*[:：]\s*", "사전 ", text)
    match = re.fullmatch(r"(?:사전|단어)\s+(.+)", text)
    query = match[1] if match else None
    if query is None:
        match = re.fullmatch(
            r"(.+?)\s*(?:뜻|의미|정의)(?:이|가|는|을)?(?:\s*(?:뭐야|뭔가요|무엇인가요|알려\s*줘|알려\s*주세요|설명해\s*줘|설명해\s*주세요))?",
            text,
        )
        if match:
            query = match[1]
    if query is None:
        match = re.fullmatch(
            r"(.+?)(?:이란|란)(?:\s*(?:뭐야|무엇인가요|무엇이야))?", text
        )
        if match:
            query = match[1]
    if query is None:
        match = re.fullmatch(r"(.+?)(?:이|가|은|는)\s*(?:뭐야|뭔가요|무엇인가요)", text)
        if match:
            query = match[1]
    if selected and query is None:
        query = re.sub(
            r"\s*(?:찾아\s*줘|찾아\s*주세요|검색해\s*줘|검색해\s*주세요)$", "", text
        )
    if query is None:
        return None
    query = re.sub(r"([\'\"”’])의$", r"\1", query.strip())
    query = query.strip(" \t\n'\"“”‘’")
    if not query or len(query) > 80:
        return None
    return query, selected


def parse_response(body, provider):
    root = ET.fromstring(body)
    code = root.findtext(".//error_code")
    if code is not None:
        if code.strip() in ("020", "021"):
            raise ValueError("인증 키가 유효하지 않거나 사용이 중지되어 있어요.")
        raise ValueError(
            "사전 API가 요청을 처리하지 못했어요. 사용 한도와 설정을 확인해 주세요."
        )
    channel = root if root.tag == "channel" else root.find("channel")
    if channel is None or channel.find("total") is None:
        raise ValueError("사전 응답 형식을 확인할 수 없어요.")
    entries = []
    for item in channel.findall("item"):
        word = (item.findtext("word") or "").strip()
        for sense in item.findall("sense"):
            definition = (sense.findtext("definition") or "").strip()
            if not word or not definition:
                continue
            link = (sense.findtext("link") or item.findtext("link") or "").strip()
            parsed = urllib.parse.urlsplit(link)
            # 인증 키가 포함된 API 주소나 외부 도메인을 출처로 전달하지 않는다.
            if (
                parsed.hostname != provider.host
                or parsed.path.startswith("/api/")
                or parsed.scheme not in ("http", "https")
                or parsed.username
                or parsed.password
            ):
                link = ""
            elif any(
                name.lower() == "key"
                for name, _ in urllib.parse.parse_qsl(parsed.query)
            ):
                link = ""
            else:
                link = urllib.parse.urlunsplit(parsed._replace(scheme="https"))
            entries.append(
                {
                    "dictionary": provider.name,
                    "word": word,
                    "definition": definition,
                    "pos": (
                        sense.findtext("pos") or item.findtext("pos") or ""
                    ).strip(),
                    "url": link,
                }
            )
            if len(entries) == 3:
                return entries
    return entries


class Dictionaries:
    def __init__(self, keys=None, opener=None):
        self.keys = (
            keys
            if keys is not None
            else {p.code: os.getenv(p.env, "").strip() for p in PROVIDERS}
        )
        self.opener = opener or open_dictionary

    @property
    def enabled(self):
        return any(self.keys.get(p.code) for p in PROVIDERS)

    def search_provider(self, provider, query):
        key = self.keys.get(provider.code)
        if not key:
            return [], f"{provider.name}: 서버에 {provider.env}를 설정해 주세요."
        params = urllib.parse.urlencode(
            {
                "key": key,
                "q": query,
                "req_type": "xml",
                "num": 10,
                "advanced": "y",
                "target": 1,
                "method": "exact",
            }
        )
        request = urllib.request.Request(
            provider.endpoint + "?" + params,
            headers={"User-Agent": "EunsuDictionary/1.0"},
        )
        try:
            with self.opener(request, timeout=6) as response:
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    raise ValueError("사전 응답이 너무 커서 표시하지 못했어요.")
            return parse_response(body, provider), None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, ssl.SSLCertVerificationError):
                return (
                    [],
                    f"{provider.name}: HTTPS 인증서 검증에 실패했어요. Python 인증서 또는 SSL_CERT_FILE 설정을 확인해 주세요.",
                )
            return (
                [],
                f"{provider.name}: 연결하지 못했어요. 잠시 후 다시 시도해 주세요.",
            )
        except (ET.ParseError, UnicodeError):
            return [], f"{provider.name}: 올바른 사전 응답을 받지 못했어요."
        except ValueError as error:
            return [], f"{provider.name}: {error}"

    def lookup(self, query, selected=None):
        providers = [p for p in PROVIDERS if selected is None or p.code == selected]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(lambda p: self.search_provider(p, query), providers)
            )
        entries, notices = [], []
        for provider, (found, error) in zip(providers, results):
            entries.extend(found)
            if error:
                notices.append(error)
            elif not found:
                notices.append(f"{provider.name}: 일치하는 표제어를 찾지 못했어요.")
        if entries:
            parts = [f"“{query}”의 사전 뜻을 찾아봤어."]
            for provider in providers:
                found = [e for e in entries if e["dictionary"] == provider.name]
                if found:
                    parts.append(
                        provider.name
                        + "\n"
                        + "\n".join(
                            f'{i}. {e["word"]}'
                            + (f' [{e["pos"]}]' if e["pos"] else "")
                            + f' — {e["definition"]}'
                            for i, e in enumerate(found, 1)
                        )
                    )
        else:
            parts = [f"“{query}”의 뜻을 확인하지 못했어."]
        if notices:
            parts.append("\n".join(notices))
        return {
            "content": "\n\n".join(parts),
            "mode": "dictionary",
            "source": "dictionary",
            "sources": entries,
        }
