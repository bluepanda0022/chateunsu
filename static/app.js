const $ = (s) => document.querySelector(s);
const STORAGE_KEY = "hangyeol-chats";
const MAX_PROMPT_HEIGHT = 150;
const MAX_FILE_BYTES = 50000;

let sessions = loadSessions();
let current = null;
let busy = false;
let analysis = null;
let toastTimer;
let selectedTraceId = null;
let tokenDraft = null;
let activeTokenMode = "model";
let tokenRequestSequence = 0;

// 브라우저 저장소
function isValidSession(session) {
  return (
    session &&
    typeof session.id === "string" &&
    Array.isArray(session.messages) &&
    session.messages.every(
      (message) =>
        message &&
        ["user", "assistant"].includes(message.role) &&
        typeof message.content === "string",
    )
  );
}

function loadSessions() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved.filter(isValidSession) : [];
  } catch {
    return [];
  }
}

function toast(text) {
  $("#toast").textContent = text;
  $("#toast").style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").style.display = "none"), 3000);
}

function saveSessions() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    toast("저장 공간이 부족해 대화를 저장하지 못했어요.");
  }
  renderHistory();
}

function renderHistory() {
  $("#history").replaceChildren();
  if (!sessions.length) {
    const emptyMessage = document.createElement("p");
    emptyMessage.className = "empty-history";
    emptyMessage.textContent = "새로운 이야기를 시작해 보세요";
    $("#history").append(emptyMessage);
  }
  sessions
    .slice()
    .reverse()
    .forEach((session) => {
      const button = document.createElement("button");
      button.textContent = session.messages[0]?.content || "새로운 대화";
      button.onclick = () => {
        if (busy) return;
        current = session;
        renderMessages();
        switchView("chat");
      };
      $("#history").append(button);
    });
}

function renderMessages() {
  $("#messages").replaceChildren();
  const messages = current?.messages || [];
  $("#welcome").classList.toggle("hidden", !!messages.length);
  messages.forEach((message, index) => {
    const row = document.createElement("div");
    row.className = "message " + message.role;
    if (message.role === "assistant") {
      const logo = document.createElement("span");
      logo.className = "logo";
      logo.textContent = "ㅇ";
      row.append(logo);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    if (message.role === "assistant") {
      const copyButton = document.createElement("button");
      copyButton.className = "copy";
      copyButton.textContent = "복사";
      copyButton.onclick = async () => {
        try {
          await navigator.clipboard.writeText(message.content);
          toast("답변을 복사했어요.");
        } catch {
          toast("이 브라우저에서 복사할 수 없어요.");
        }
      };
      bubble.append(copyButton);
      if (message.token_trace_id) {
        const inspect = document.createElement("button");
        inspect.className = "teach-button";
        inspect.textContent = "실제 추론 토큰 보기";
        inspect.onclick = () => {
          selectedTraceId = message.token_trace_id;
          prepareTokenMode("inference-input");
          $("#tokenRound").replaceChildren();
          switchView("tokens");
          analyzeTokens();
        };
        bubble.append(inspect);
      }
      if (
        Array.isArray(message.dictionary_lookups) &&
        message.dictionary_lookups.length
      ) {
        const info = document.createElement("small");
        info.className = "lookup-info";
        info.textContent =
          "모델이 사전 확인: " +
          message.dictionary_lookups
            .map((item) => item.word + (item.found ? "" : " (결과 없음)"))
            .join(", ");
        bubble.append(info);
      }
      if (Array.isArray(message.sources)) {
        const citations = document.createElement("div");
        citations.className = "dictionary-sources";
        for (const source of message.sources) {
          if (!source || typeof source.url !== "string") continue;
          try {
            const url = new URL(source.url);
            if (
              url.protocol !== "https:" ||
              !["stdict.korean.go.kr", "opendict.korean.go.kr"].includes(
                url.hostname,
              )
            )
              continue;
            const link = document.createElement("a");
            link.href = url.href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = `${source.dictionary} · ${source.word} ↗`;
            citations.append(link);
          } catch {
            /* 이전 기록의 잘못된 출처는 생략한다. */
          }
        }
        bubble.append(citations);
      }
      const question = messages[index - 1];
      if (question?.role === "user") {
        const teachButton = document.createElement("button");
        teachButton.className = "teach-button";
        teachButton.textContent = "답변 가르치기";
        teachButton.onclick = () => {
          $("#teachQuestion").value = question.content;
          $("#teachAnswer").value = "";
          $("#teachStatus").textContent = "";
          $("#teachDialog").showModal();
          $("#teachAnswer").focus();
        };
        bubble.append(teachButton);
      }
    }
    row.append(bubble);
    $("#messages").append(row);
  });
}

function switchView(name) {
  $("#chatView").classList.toggle("hidden", name !== "chat");
  $("#tokensView").classList.toggle("hidden", name !== "tokens");
  $("#pageTitle").textContent = {
    chat: "은수 채팅",
    tokens: "토크나이저 실험실",
  }[name];
  document
    .querySelectorAll(".nav")
    .forEach((n) => n.classList.toggle("active", n.dataset.view === name));
  $("#sidebar").classList.remove("open");
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw Error(
      "로컬 서버에 연결할 수 없어요. python3 app.py로 실행해 주세요.",
    );
  }
  if (!res.headers.get("Content-Type")?.includes("application/json")) {
    throw Error(
      "서버가 올바른 응답을 보내지 않았어요. 기존 서버를 종료하고 python3 app.py로 다시 실행해 주세요.",
    );
  }
  const data = await res.json();
  if (!res.ok) throw Error(data.error || "요청에 실패했어요.");
  return data;
}

function resizePrompt() {
  const p = $("#prompt");
  p.style.height = "auto";
  p.style.height = Math.min(p.scrollHeight, MAX_PROMPT_HEIGHT) + "px";
  $("#charCount").textContent = p.value.length + "자";
}

async function streamChat(messages, onUpdate) {
  const history = messages
    .slice(-16)
    .map(({ role, content, source, mode, topics, dialogue, style }) => ({
      role,
      content,
      source,
      mode,
      topics,
      dialogue,
      style,
    }));
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw Error(error.error || "답변 요청에 실패했어요.");
  }
  if (!response.headers.get("Content-Type")?.includes("application/x-ndjson"))
    throw Error("서버를 재시작하고 페이지를 새로고침해 주세요.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "",
    content = "",
    result;
  function accept(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "error") throw Error(event.error);
    if (event.type === "status") onUpdate(event.content);
    if (event.type === "reset") {
      content = "";
      onUpdate("은수가 확인하고 있어요…");
    }
    if (event.type === "delta") {
      content += event.content;
      onUpdate(content);
    }
    if (event.type === "done") result = event.response;
  }
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      lines.forEach(accept);
      if (done) break;
    }
    if (buffer.trim()) accept(buffer);
    if (!result) throw Error("답변 생성이 중단됐어요. 다시 시도해 주세요.");
    return result;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function submitChat(e) {
  e.preventDefault();
  const content = $("#prompt").value.trim();
  if (!content || busy) return;
  busy = true;
  $("#send").disabled = true;
  if (!current) {
    current = { id: crypto.randomUUID(), messages: [] };
    sessions.push(current);
  }
  const target = current;
  target.messages.push({ role: "user", content });
  $("#prompt").value = "";
  resizePrompt();
  renderMessages();
  saveSessions();
  const loading = document.createElement("div");
  loading.className = "message assistant";
  loading.textContent = "은수가 생각을 정리하고 있어요…";
  $("#messages").append(loading);
  loading.scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    const data = await streamChat(target.messages, (text) => {
      loading.textContent = text;
    });
    target.messages.push({
      role: "assistant",
      content: data.content,
      source: data.source,
      sources: data.sources,
      topics: data.topics,
      style: data.style,
      mode: data.mode,
      dialogue: data.dialogue,
      model: data.model,
      usage: data.usage,
      token_trace_id: data.token_trace_id,
      dictionary_lookups: data.dictionary_lookups,
    });
    $("#modeLabel").textContent =
      data.mode === "dictionary"
        ? data.style === "conversation"
          ? "사전 기반 대화"
          : "국어사전 검색"
        : data.mode === "model"
          ? `로컬 AI · ${data.model}`
          : "규칙 기반 대화";
    saveSessions();
  } catch (err) {
    target.messages.pop();
    $("#prompt").value = content;
    resizePrompt();
    saveSessions();
    toast(err.message);
  } finally {
    busy = false;
    $("#send").disabled = false;
    renderMessages();
    $("#chatForm").scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

function newChat() {
  if (busy) return;
  current = null;
  selectedTraceId = null;
  renderMessages();
  switchView("chat");
  $("#prompt").focus();
}

function clearHistory() {
  if (busy) return;
  if (sessions.length && confirm("이 브라우저의 모든 대화를 삭제할까요?")) {
    sessions = [];
    current = null;
    saveSessions();
    renderMessages();
  }
}

async function importTextFile(e) {
  const f = e.target.files[0];
  if (!f) return;
  if (f.size > MAX_FILE_BYTES) {
    toast("50KB 이하의 텍스트 파일을 선택해 주세요.");
    return;
  }
  $("#prompt").value = await f.text();
  resizePrompt();
  toast("파일 내용을 입력창에 불러왔어요.");
  e.target.value = "";
}

function prepareTokenMode(mode) {
  const input = $("#tokenInput");
  const wasInference = activeTokenMode.startsWith("inference-");
  const inference = mode.startsWith("inference-");
  if (!wasInference) tokenDraft = input.value;
  if (wasInference && !inference) input.value = tokenDraft ?? "";
  input.readOnly = inference;
  $("#tokenMode").value = mode;
  $("#tokenRound").classList.toggle("hidden", !inference);
  activeTokenMode = mode;
}

async function analyzeTokens() {
  const mode = $("#tokenMode").value;
  prepareTokenMode(mode);
  const text = $("#tokenInput").value;
  const inference = mode.startsWith("inference-");
  if (!inference && !text.trim()) {
    toast("분석할 텍스트를 입력해 주세요.");
    return;
  }
  $("#analyze").disabled = true;
  $("#tokenMode").disabled = true;
  const requestSequence = ++tokenRequestSequence;
  analysis = null;
  $("#tokenCount").textContent = "—";
  $("#tokenChars").textContent = "—";
  $("#tokenBytes").textContent = "—";
  $("#tokenOutput").textContent = "분석 중…";
  $("#tokenVerification").textContent = "";
  try {
    const last = current?.messages
      .slice()
      .reverse()
      .find((m) => m.token_trace_id);
    const traceId = selectedTraceId || last?.token_trace_id;
    if (inference && !traceId)
      throw Error(
        "먼저 은수와 대화한 뒤 답변 아래 ‘실제 추론 토큰 보기’를 눌러 주세요.",
      );
    const round = $("#tokenRound").value;
    const result = await api("/api/tokenize", {
      text,
      mode,
      trace_id: traceId,
      ...(round !== "" ? { round_index: Number(round) } : {}),
    });
    if (requestSequence !== tokenRequestSequence) return;
    analysis = result;
    $("#tokenInput").readOnly = inference;
    $("#tokenRound").classList.toggle("hidden", !inference);
    if (inference) {
      $("#tokenInput").value = analysis.source_text;
      $("#tokenRound").replaceChildren();
      for (let i = 0; i < analysis.rounds; i++) {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `${i + 1}차 · ${analysis.round_kinds?.[i] === "dictionary-decision" ? "사전 판단" : "답변 생성"}`;
        $("#tokenRound").append(option);
      }
      $("#tokenRound").value = analysis.round_index;
      $("#tokenVerification").textContent =
        `실행 모델: ${analysis.model} · 어휘 검증 ${analysis.verified ? "완료" : "미확인"} · 백엔드 입력 ${analysis.reported_input_tokens}, 출력 ${analysis.reported_output_tokens} 토큰. 입력 수 일치: ${analysis.input_count_matches ? "예" : "아니오"} · 백엔드 입력 ID 일치: ${analysis.context_prefix_matches ? "예" : "미확인"}. 출력 목록은 모델 원문의 BPE 재인코딩이며 샘플링 순서의 원본 ID 목록은 아닙니다. 백엔드 출력 수에는 종료 신호 등이 포함되어 차이가 날 수 있습니다.`;
    } else {
      $("#tokenVerification").textContent = analysis.verified
        ? "현재 실행 모델의 어휘와 ID를 대조했습니다. 입력 본문만 분석하며 채팅 전체가 아닙니다."
        : "한국어 형태소 분석은 모델의 BPE 토큰과 별도입니다.";
    }
    $("#tokenCountLabel").textContent =
      mode === "morpheme" ? "형태소 수" : "모델 토큰 수";
    $("#tokenScope").textContent = {
      morpheme: "입력 본문 · 형태소 분석",
      model: "입력 본문만 · 시스템 지침·대화 이력 제외",
      "inference-input": "전체 추론 입력 · 시스템 지침·대화 이력 포함",
      "inference-output": "답변 본문만 · 재인코딩",
    }[mode];
    const fragmentCount = analysis.tokens.filter((t) => t.byte_fragment).length;
    if (fragmentCount)
      $("#tokenVerification").textContent +=
        ` 독립적으로 글자가 되지 않는 바이트 토큰 ${fragmentCount}개는 ⟨16진수 바이트⟩로 표시합니다. 각각 실제 토큰 1개입니다.`;
    $("#tokenMethod").textContent = analysis.method;
    $("#tokenCount").textContent = analysis.tokens.length;
    $("#tokenChars").textContent = analysis.characters;
    $("#tokenBytes").textContent = analysis.bytes;
    $("#tokenOutput").replaceChildren();
    analysis.tokens.forEach((t, index) => {
      const s = document.createElement("span");
      s.className = "token";
      const display = t.byte_fragment ? `⟨${t.byte_hex}⟩` : t.text;
      s.textContent =
        (t.byte_fragment
          ? display
          : display
              .replaceAll(" ", "·")
              .replaceAll("\n", "↵")
              .replaceAll("\t", "⇥")
              .replaceAll("\r", "␍")) + (t.pos ? ` / ${t.pos}` : "");
      s.title =
        `#${index + 1} · ${t.pos ? "품사: " + t.pos : "실행 모델 ID: " + t.id} · 원문 범위: ${t.surface} · 위치 ${t.start}~${t.end}` +
        (t.byte_hex ? ` · 실제 바이트: ${t.byte_hex}` : "") +
        (t.byte_fragment
          ? " · 한 글자를 여러 토큰이 나누어 표현하므로 원문 범위가 겹칠 수 있습니다."
          : "");
      if (t.id !== undefined) {
        const id = document.createElement("small");
        id.className = "token-id";
        id.textContent = `#${index + 1} · ${t.id}`;
        s.append(id);
      }
      $("#tokenOutput").append(s);
    });
  } catch (e) {
    if (requestSequence !== tokenRequestSequence) return;
    $("#tokenOutput").textContent = e.message;
    toast(e.message);
  } finally {
    if (requestSequence === tokenRequestSequence) {
      $("#analyze").disabled = false;
      $("#tokenMode").disabled = false;
    }
  }
}

function exportTokens() {
  if (!analysis) {
    toast("먼저 토큰을 분석해 주세요.");
    return;
  }
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "eunsu-tokens.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function teachReply(event) {
  event.preventDefault();
  const button = $("#saveTeaching");
  button.disabled = true;
  $("#teachStatus").textContent = "";
  try {
    const result = await api("/api/teach", {
      question: $("#teachQuestion").value,
      answer: $("#teachAnswer").value,
    });
    $("#teachDialog").close();
    toast(
      result.saved ? "배웠어! 같은 말을 다시 해 봐." : "이미 배운 답변이에요.",
    );
  } catch (error) {
    $("#teachStatus").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

// 이벤트 연결 및 초기 화면
function bindEvents() {
  $("#prompt").addEventListener("input", resizePrompt);
  $("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("#chatForm").requestSubmit();
    }
  });
  $("#newChat").onclick = newChat;
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      newChat();
    }
  });
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => (b.onclick = () => switchView(b.dataset.view)));
  document.querySelectorAll("[data-prompt]").forEach(
    (b) =>
      (b.onclick = () => {
        $("#prompt").value = b.dataset.prompt;
        resizePrompt();
        $("#prompt").focus();
      }),
  );
  $("#menu").onclick = () => $("#sidebar").classList.toggle("open");
  $("#info").onclick = $("#about").onclick = () =>
    $("#aboutDialog").showModal();
  $("#closeDialog").onclick = () => $("#aboutDialog").close();
  $("#attach").onclick = () => $("#fileInput").click();
  $("#sample").onclick = () => {
    prepareTokenMode("model");
    $("#tokenInput").value = "한국어는 아름다운 언어예요. Hello, 은수! 👋";
    analyzeTokens();
  };

  $("#teachForm").onsubmit = teachReply;
  $("#cancelTeaching").onclick = () => $("#teachDialog").close();
  $("#chatForm").onsubmit = submitChat;
  $("#clearHistory").onclick = clearHistory;
  $("#fileInput").onchange = importTextFile;
  $("#analyze").onclick = analyzeTokens;
  $("#tokenMode").onchange = () => {
    $("#tokenRound").replaceChildren();
    analyzeTokens();
  };
  $("#tokenRound").onchange = analyzeTokens;
  $("#exportTokens").onclick = exportTokens;
}

bindEvents();
renderHistory();
renderMessages();
