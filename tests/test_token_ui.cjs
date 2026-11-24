// 실행: node --test tests/test_token_ui.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(value = "") {
    this.value = value;
    this.children = [];
    this.classList = { toggle() {} };
    this.style = {};
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get textContent() {
    return this.text || "";
  }
  replaceChildren() {
    this.children = [];
    this.text = "";
  }
  append(child) {
    this.children.push(child);
  }
  addEventListener() {}
}

function setup() {
  const elements = new Map();
  const get = (selector) => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  get("#tokenInput").value = "안녕하세요, 은수와 함께 한국어를 탐험해요!";
  get("#tokenMode").value = "model";
  const context = vm.createContext({
    document: {
      querySelector: get,
      createElement: () => new Element(),
      querySelectorAll: () => [],
      addEventListener() {},
    },
    localStorage: { getItem: () => null },
    setTimeout: () => 1,
    clearTimeout() {},
    calls: [],
  });
  const source = fs.readFileSync(
    require("node:path").join(__dirname, "../static/app.js"),
    "utf8",
  );
  vm.runInContext(
    source.replace(
      /bindEvents\(\);\s*renderHistory\(\);\s*renderMessages\(\);\s*$/,
      "",
    ),
    context,
  );
  vm.runInContext(
    `api = async (path, body) => {
    calls.push(body);
    if (globalThis.fail) throw Error("연결 실패");
    return {
      tokens: [{text: " �", id: 16751, byte_fragment: true, byte_hex: "20 EC 9D", surface: " 은", start: 0, end: 2}],
      method: "BPE", source_text: body.mode.startsWith("inference-") ? "SYSTEM + HISTORY + USER" : body.text,
      characters: body.text.length, bytes: 10, round_index: 0, rounds: 1, round_kinds: ["answer"], verified: true
    };
  }; selectedTraceId = "trace";`,
    context,
  );
  return { get, context, run: (code) => vm.runInContext(code, context) };
}

test("전체 입력과 출력 사이를 이동한 뒤 본문 모드로 돌아오면 원래 문장을 분석한다", async () => {
  const { get, context, run } = setup();
  const original = get("#tokenInput").value;
  await run("analyzeTokens()");
  get("#tokenMode").value = "inference-input";
  await run("analyzeTokens()");
  assert.equal(get("#tokenInput").value, "SYSTEM + HISTORY + USER");
  assert.equal(get("#tokenInput").readOnly, true);
  get("#tokenMode").value = "inference-output";
  await run("analyzeTokens()");
  get("#tokenMode").value = "model";
  await run("analyzeTokens()");
  assert.equal(get("#tokenInput").value, original);
  assert.equal(context.calls.at(-1).text, original);
  assert.equal(get("#tokenInput").readOnly, false);
});

test("바이트 조각도 ID당 상자 하나로 표시하고 실패 시 이전 수치를 남기지 않는다", async () => {
  const { get, context, run } = setup();
  await run("analyzeTokens()");
  const chips = get("#tokenOutput").children;
  assert.equal(chips.length, 1);
  assert.equal(chips[0].textContent, "⟨20 EC 9D⟩");
  assert.equal(chips[0].children[0].textContent, "#1 · 16751");
  assert.equal(get("#tokenCount").textContent, "1");
  context.fail = true;
  await run("analyzeTokens()");
  assert.equal(get("#tokenCount").textContent, "—");
  assert.equal(get("#tokenOutput").textContent, "연결 실패");
  assert.equal(run("analysis"), null);
});
