"""Parse k-shot plaintext prompts into multi-turn chat messages."""

from __future__ import annotations

import re
from typing import Optional


def _split_blocks(text: str, start_markers: list[str]) -> tuple[str, list[str]]:
    """Split text into blocks starting with any start marker.

    Returns (prefix, blocks). prefix is any leading text before the first marker.
    """
    if not text:
        return "", []

    pattern = r"(?m)^(" + "|".join(re.escape(m) for m in start_markers) + r")"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text.strip(), []

    prefix = text[:matches[0].start()].strip()
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())
    return prefix, blocks


def _split_user_assistant(block: str, answer_markers: list[str]) -> tuple[str, Optional[str]]:
    """Split a block into (user, assistant) at the first answer marker.

    The answer marker stays with the user content to preserve original format.
    Returns (user, assistant) where assistant may be None.
    """
    pattern = r"(?m)^(" + "|".join(re.escape(m) for m in answer_markers) + r")"
    match = re.search(pattern, block)
    if not match:
        return block.strip(), None

    split_at = match.end()
    user = block[:split_at].rstrip()
    assistant = block[split_at:].lstrip()
    if not assistant:
        return user, None
    return user, assistant


def parse_kshot_messages(text: str, dataset_name: str) -> Optional[list[dict]]:
    """Parse k-shot prompt into chat messages, if possible.

    Returns a list of {"role": ..., "content": ...} or None if parsing fails.
    """
    name = (dataset_name or "").lower()
    if name == "gsm8k":
        start_markers = ["Q:", "Question:"]
        answer_markers = ["A:", "Answer:"]
    elif name == "math":
        start_markers = ["Problem:"]
        answer_markers = ["Solution:"]
    else:
        return None

    prefix, blocks = _split_blocks(text, start_markers)
    if not blocks:
        return None

    messages: list[dict] = []
    parsed_any_assistant = False
    parse_failed = False

    for i, block in enumerate(blocks):
        user, assistant = _split_user_assistant(block, answer_markers)
        if i == 0 and prefix:
            user = prefix + "\n\n" + user
        messages.append({"role": "user", "content": user})

        if assistant is None:
            # Allow missing assistant only on the last block (final question).
            if i != len(blocks) - 1:
                parse_failed = True
                break
            continue

        messages.append({"role": "assistant", "content": assistant})
        parsed_any_assistant = True

    if parse_failed or not parsed_any_assistant:
        return None

    return messages

