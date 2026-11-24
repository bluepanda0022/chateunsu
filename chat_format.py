"""Qwen3의 채팅 형식을 한 곳에서 렌더링한다."""


def wrap_messages(messages, system, model=None):
    prompt = "<|im_start|>system\n" + system + "<|im_end|>\n"
    for message in messages:
        if message["role"] in ("user", "assistant"):
            prompt += (
                "<|im_start|>"
                + message["role"]
                + "\n"
                + message["content"]
                + "<|im_end|>\n"
            )
    return prompt + "<|im_start|>assistant\n"
