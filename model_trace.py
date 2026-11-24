"""실행 모델과 어휘를 대조하고 정확히 렌더링한 입력의 BPE ID를 제공한다."""

import json

from korean_tokenizer import describe_model_tokens, model_tokenizer


def verify_vocabulary(info, model="qwen3"):
    tokens = info.get("model_info", {}).get("tokenizer.ggml.tokens", [])
    vocab = model_tokenizer(model).get_vocab()
    tokenizer = model_tokenizer(model)
    merges = json.loads(tokenizer.to_str())["model"].get("merges", [])
    expected = [" ".join(pair) if isinstance(pair, list) else pair for pair in merges]
    return (
        bool(tokens)
        and all(i < len(tokens) and tokens[i] == piece for piece, i in vocab.items())
        and info.get("model_info", {}).get("tokenizer.ggml.merges") == expected
    )


def input_tokens(prompt, model="qwen3"):
    return describe_model_tokens(prompt, model_tokenizer(model))
