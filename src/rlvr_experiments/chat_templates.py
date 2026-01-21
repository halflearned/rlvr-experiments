"""Chat template helpers for training/eval parity.

These templates mirror implementations in external/olmes so we can apply the
same formatting during training runs.
"""

from __future__ import annotations


def create_prompt_with_tulu_thinker_r1_style_chat_format(
    messages, tokenizer=None, eos="</s>", add_bos=True, add_generation_prompt=True
):
    """Format messages with the Tulu thinker R1-style template.

    Matches external/olmes/oe_eval/tasks/chat_templates.py.
    """
    del tokenizer, add_bos  # Unused but kept for signature compatibility.

    formatted_text = (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>.\n\n"
    )

    num_messages = len(messages)
    for i, message in enumerate(messages):
        is_last_iteration = i == num_messages - 1
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            formatted_text += "<|system|>\n" + content + "\n"
        elif role == "user":
            formatted_text += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            if "</think>" in content:
                content = content.split("</think>")[-1]
            formatted_text += "<|assistant|>\n" + content
            if not is_last_iteration:
                formatted_text += eos + "\n"
            else:
                formatted_text += eos
        else:
            raise ValueError(
                "Tulu thinker r1 style chat template only supports 'system', "
                "'user' and 'assistant' roles. Invalid role: {}.".format(role)
            )

        if is_last_iteration and add_generation_prompt:
            formatted_text += "<|assistant|>\n<think>"

    return formatted_text


CHAT_TEMPLATES = {
    "tulu_thinker_r1_style": create_prompt_with_tulu_thinker_r1_style_chat_format,
}

