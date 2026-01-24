"""Verifier for IF_multi_constraints_upto5 instruction-following dataset.

Implements a lightweight subset of AllenAI open-instruct IFEvalG instruction
checks, adapted to our codebase without external dependencies (e.g., nltk).
"""

from __future__ import annotations

import ast
import json
import re
import time
from collections import Counter
from typing import Any

import langdetect


_COMPARISON_RELATION = ("less than", "at least")
_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|"
    r"But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def _remove_thinking_section(text: str) -> str:
    text = text.replace("<|assistant|>", "").strip()
    text = text.split("</think>")[-1]
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def _word_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _count_words(text: str) -> int:
    return len(_word_tokenize(text))


def _count_sentences(text: str) -> int:
    return len(_split_into_sentences(text))


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def _keywords_existence(text: str, keywords: list[str]) -> bool:
    return all(re.search(keyword, text, flags=re.IGNORECASE) for keyword in keywords)


def _keywords_frequency(text: str, keyword: str, frequency: int, relation: str) -> bool:
    keyword = _as_str(keyword).strip()
    if not keyword:
        return False
    actual = len(re.findall(keyword, text, flags=re.IGNORECASE))
    if relation == _COMPARISON_RELATION[0]:
        return actual < frequency
    if relation == _COMPARISON_RELATION[1]:
        return actual >= frequency
    return False


def _keywords_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    return all(not re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE) for word in forbidden_words)


def _keywords_word_once(text: str, keyword: str) -> bool:
    keyword = _as_str(keyword).strip()
    if not keyword:
        return False
    return len(re.findall(keyword, text, flags=re.IGNORECASE)) == 1


def _keywords_letter_frequency(text: str, letter: str, let_frequency: int, let_relation: str) -> bool:
    letter = _as_str(letter).lower()
    if len(letter) != 1:
        return False
    letters = Counter(text.lower())
    if let_relation == _COMPARISON_RELATION[0]:
        return letters[letter] < let_frequency
    if let_relation == _COMPARISON_RELATION[1]:
        return letters[letter] >= let_frequency
    return False


def _keywords_no_adjacent_consecutive(text: str) -> bool:
    words = text.split()
    for i in range(len(words) - 1):
        first_letter = words[i][:1].lower()
        second_letter = words[i + 1][:1].lower()
        if len(first_letter) != 1 or len(second_letter) != 1:
            return False
        if ord(second_letter) - ord(first_letter) == 1:
            return False
    return True


def _keywords_palindrome(text: str) -> bool:
    return any(word == word[::-1] for word in text.split())


def _keywords_start_end(text: str) -> bool:
    words = _word_tokenize(text)
    if len(words) < 2:
        return False
    return words[0].lower() == words[-1].lower()


def _keywords_exclude_word_harder(text: str, keyword: str) -> bool:
    keyword = _as_str(keyword).strip()
    if not keyword:
        return False
    return f" {keyword} " not in text


def _keywords_specific_position(text: str, keyword: str, n: int, m: int) -> bool:
    keyword = _as_str(keyword).strip()
    if not keyword:
        return False
    sentences = _split_into_sentences(text)
    if len(sentences) < n:
        return False
    words = _word_tokenize(sentences[n - 1])
    if len(words) < m:
        return False
    return words[m - 1] == keyword


def _language_response_language(text: str, language: str) -> bool:
    try:
        return langdetect.detect(text) == language
    except langdetect.LangDetectException:
        return True


def _length_number_sentences(text: str, num_sentences: int, relation: str) -> bool:
    actual = _count_sentences(text)
    if relation == _COMPARISON_RELATION[0]:
        return actual < num_sentences
    if relation == _COMPARISON_RELATION[1]:
        return actual >= num_sentences
    return False


def _length_number_words(text: str, num_words: int, relation: str) -> bool:
    actual = _count_words(text)
    if relation == _COMPARISON_RELATION[0]:
        return actual < num_words
    if relation == _COMPARISON_RELATION[1]:
        return actual >= num_words
    return False


def _length_number_paragraphs(text: str, num_paragraphs: int) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


def _length_nth_paragraph_first_word(text: str, num_paragraphs: int, nth_paragraph: int, first_word: str) -> bool:
    first_word = _as_str(first_word).lower()
    paragraphs = re.split(r"\n\n", text)
    count = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            count -= 1
    if nth_paragraph > count or nth_paragraph <= 0:
        return False
    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    extracted = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        extracted += letter.lower()
    return count == num_paragraphs and extracted == first_word


def _detectable_content_placeholders(text: str, num_placeholders: int) -> bool:
    placeholders = re.findall(r"\[.*?\]", text)
    return len(placeholders) >= num_placeholders


def _detectable_content_postscript(text: str, postscript_marker: str) -> bool:
    value = text.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _detectable_format_number_bullet_lists(text: str, num_bullets: int) -> bool:
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
    return len(bullet_lists) + len(bullet_lists_2) == num_bullets


def _detectable_format_constrained_response(text: str) -> bool:
    value = text.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSE_OPTIONS)


def _detectable_format_number_highlighted_sections(text: str, num_highlights: int) -> bool:
    highlights = re.findall(r"\*[^\n\*]*\*", text)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", text)
    count = 0
    for highlight in highlights:
        if highlight.strip("*").strip():
            count += 1
    for highlight in double_highlights:
        stripped = highlight[2:-2].strip()
        if stripped:
            count += 1
    return count >= num_highlights


def _detectable_format_multiple_sections(text: str, section_spliter: str, num_sections: int) -> bool:
    section_splitter_pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
    sections = re.split(section_splitter_pattern, text)
    return (len(sections) - 1) >= num_sections


def _detectable_format_json(text: str) -> bool:
    value = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _detectable_format_title(text: str) -> bool:
    return bool(re.findall(r"<<(.*?)>>", text))


def _detectable_format_sentence_hyphens(text: str) -> bool:
    sentences_gold = re.sub("-", " ", text)
    sentences_gold = _split_into_sentences(sentences_gold)
    sentences = text.split("-")
    for sentence, gold in zip(sentences, sentences_gold):
        if sentence.strip() != sentence or sentence != gold:
            return False
    return True


def _detectable_format_square_brackets(text: str) -> bool:
    words = text.split()
    return all(word.startswith("[") and word.endswith("]") for word in words)


def _detectable_format_bigram_wrapping(text: str) -> bool:
    words = text.split()
    for i in range(0, len(words) - 1, 2):
        if i + 1 < len(words) and not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
            return False
    return True


def _combination_two_responses(text: str) -> bool:
    valid = []
    responses = text.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _combination_repeat_prompt(text: str, prompt_to_repeat: str) -> bool:
    prompt_to_repeat = _as_str(prompt_to_repeat).strip()
    if not prompt_to_repeat:
        return False
    return text.strip().lower().startswith(prompt_to_repeat.lower())


def _startend_end_checker(text: str, end_phrase: str) -> bool:
    value = text.strip().strip('"').lower()
    end_phrase = _as_str(end_phrase).strip().lower()
    if not end_phrase:
        return False
    return value.endswith(end_phrase)


def _startend_quotation(text: str) -> bool:
    value = text.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


def _change_case_capital_word_frequency(text: str, capital_frequency: int, capital_relation: str) -> bool:
    words = _word_tokenize(text)
    capital_words = len([word for word in words if word.isupper()])
    if capital_relation == _COMPARISON_RELATION[0]:
        return capital_words < capital_frequency
    return capital_words >= capital_frequency


def _change_case_english_capital(text: str) -> bool:
    try:
        return text.isupper() and langdetect.detect(text) == "en"
    except langdetect.LangDetectException:
        return True


def _change_case_english_lowercase(text: str) -> bool:
    try:
        return text.islower() and langdetect.detect(text) == "en"
    except langdetect.LangDetectException:
        return True


def _punctuation_no_comma(text: str) -> bool:
    return not re.search(r"\,", text)


def _punctuation_dot(text: str) -> bool:
    return not re.search(r"\.", text)


def _punctuation_exclamation(text: str) -> bool:
    return not re.search(r"\!", text)


def _copy_repeat_phrase(text: str, phrase: str, small_n: int) -> bool:
    phrase = _as_str(phrase).strip()
    if not phrase:
        return False
    first_word = phrase.split()[0]
    last_word = phrase.split()[-1]
    found_phrases = re.findall(rf"{re.escape(first_word)} .*? {re.escape(last_word)}", text)
    if len(found_phrases) != small_n:
        return False
    ref_phrase = phrase.split()
    for candidate in found_phrases:
        words = candidate.split()
        if len(words) != len(ref_phrase):
            return False
        differences = 0
        for a, b in zip(words, ref_phrase):
            if a != b:
                differences += 1
                if differences > 1:
                    return False
    return True


def _copy_copy(text: str, prompt_to_repeat: str) -> bool:
    prompt_to_repeat = _as_str(prompt_to_repeat).strip()
    if not prompt_to_repeat:
        return False
    return text.strip().lower() == prompt_to_repeat.lower()


def _copy_copying_multiple(text: str, prompt_to_repeat: str, N: int) -> bool:
    prompt_to_repeat = _as_str(prompt_to_repeat).strip()
    if not prompt_to_repeat:
        return False
    prompts = text.split("******")
    if len(prompts) != N:
        return False
    return all(prompt.strip().lower() == prompt_to_repeat.lower() for prompt in prompts)


def _copy_span_idx(text: str, prompt_to_repeat: str, n_start: int, n_end: int) -> bool:
    prompt_to_repeat = _as_str(prompt_to_repeat)
    if not prompt_to_repeat:
        return False
    span = prompt_to_repeat[n_start:n_end]
    return text.strip().lower() == span.strip().lower()


def _letters_letter_counting(text: str, N: int, relation: str) -> bool:
    letters = re.findall(r"[a-zA-Z]", text)
    if relation == _COMPARISON_RELATION[1]:
        return len(letters) >= N
    if relation == _COMPARISON_RELATION[0]:
        return len(letters) < N
    return False


def _letters_letter_counting2(text: str, letter: str, let_frequency: int, let_relation: str) -> bool:
    return _keywords_letter_frequency(text, letter, let_frequency, let_relation)


def _paragraphs_paragraphs(text: str) -> bool:
    return _length_number_paragraphs(text, 2)


def _paragraphs_paragraphs2(text: str) -> bool:
    paragraphs = re.split(r"\n\n", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == 2


def _first_word_sent(text: str, first_word: str) -> bool:
    first_word = _as_str(first_word).strip()
    if not first_word:
        return False
    sentences = _split_into_sentences(text)
    for sentence in sentences:
        if not sentence.strip():
            return False
        if sentence.split()[0].strip().lower() != first_word.lower():
            return False
    return True


def _first_word_answer(text: str, first_word: str) -> bool:
    first_word = _as_str(first_word).strip()
    if not first_word or not text.strip():
        return False
    return text.split()[0].strip().lower() == first_word.lower()


def _last_word_sent(text: str, last_word: str) -> bool:
    last_word = _as_str(last_word).strip()
    if not last_word:
        return False
    sentences = _split_into_sentences(text)
    for sentence in sentences:
        if not sentence.strip():
            return False
        word = sentence.split()[-1].strip()
        word = re.sub(r"[^\w\s]", "", word)
        if word.lower() != last_word.lower():
            return False
    return True


def _last_word_answer(text: str, last_word: str) -> bool:
    last_word = _as_str(last_word).strip()
    if not last_word or not text.strip():
        return False
    word = text.split()[-1].strip()
    word = re.sub(r"[^\w\s]", "", word)
    return word.lower() == last_word.lower()


def _count_lowercase(text: str, N: int) -> bool:
    lowercase_words = re.findall(r"\b[a-z]+\b", text)
    return len(lowercase_words) <= N


def _count_unique(text: str) -> bool:
    words = _word_tokenize(text)
    return len(words) == len(set(words))


def _counting_composition(text: str, n_sent: int, n_words: int) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
        sentences = _split_into_sentences(paragraph)
        if len(sentences) != n_sent:
            return False
        for sentence in sentences:
            words = _word_tokenize(sentence)
            if len(words) != n_words:
                return False
    return count == 3


def _count_increment_word(text: str, keyword1: str, keyword2: str) -> bool:
    keyword1 = _as_str(keyword1).strip()
    keyword2 = _as_str(keyword2).strip()
    if not keyword1 or not keyword2:
        return False
    actual1 = len(re.findall(keyword1, text, flags=re.IGNORECASE))
    actual2 = len(re.findall(keyword2, text, flags=re.IGNORECASE))
    return bool(actual1 == 1 and actual2 == 2)


INSTRUCTION_FUNCTIONS = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _keywords_forbidden_words,
    "keywords:letter_frequency": _keywords_letter_frequency,
    "language:response_language": _language_response_language,
    "length_constraints:number_sentences": _length_number_sentences,
    "length_constraints:number_paragraphs": _length_number_paragraphs,
    "length_constraints:number_words": _length_number_words,
    "length_constraints:nth_paragraph_first_word": _length_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _detectable_content_placeholders,
    "detectable_content:postscript": _detectable_content_postscript,
    "detectable_format:number_bullet_lists": _detectable_format_number_bullet_lists,
    "detectable_format:constrained_response": _detectable_format_constrained_response,
    "detectable_format:number_highlighted_sections": _detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": _detectable_format_multiple_sections,
    "detectable_format:json_format": _detectable_format_json,
    "detectable_format:title": _detectable_format_title,
    "combination:two_responses": _combination_two_responses,
    "combination:repeat_prompt": _combination_repeat_prompt,
    "startend:end_checker": _startend_end_checker,
    "change_case:capital_word_frequency": _change_case_capital_word_frequency,
    "change_case:english_capital": _change_case_english_capital,
    "change_case:english_lowercase": _change_case_english_lowercase,
    "punctuation:no_comma": _punctuation_no_comma,
    "startend:quotation": _startend_quotation,
    "copy:repeat_phrase": _copy_repeat_phrase,
    "copy:copy": _copy_copy,
    "new:copy_span_idx": _copy_span_idx,
    "detectable_format:sentence_hyphens": _detectable_format_sentence_hyphens,
    "keywords:no_adjacent_consecutive": _keywords_no_adjacent_consecutive,
    "detectable_format:square_brackets": _detectable_format_square_brackets,
    "keywords:word_once": _keywords_word_once,
    "keywords:word_count_different_numbers": _keywords_frequency,
    "keywords:exclude_word_harder": _keywords_exclude_word_harder,
    "paragraphs:paragraphs": _paragraphs_paragraphs,
    "paragraphs:paragraphs2": _paragraphs_paragraphs2,
    "first_word:first_word_sent": _first_word_sent,
    "first_word:first_word_answer": _first_word_answer,
    "last_word:last_word_sent": _last_word_sent,
    "last_word:last_word_answer": _last_word_answer,
    "detectable_format:bigram_wrapping": _detectable_format_bigram_wrapping,
    "copy:copying_simple": _copy_copy,
    "copy:copying_multiple": _copy_copying_multiple,
    "punctuation:punctuation_dot": _punctuation_dot,
    "punctuation:punctuation_exclamation": _punctuation_exclamation,
    "count:lowercase_counting": _count_lowercase,
    "letters:letter_counting": _letters_letter_counting,
    "letters:letter_counting2": _letters_letter_counting2,
    "count:counting_composition": _counting_composition,
    "count:count_unique": _count_unique,
    "count:count_increment_word": _count_increment_word,
    "keywords:palindrome": _keywords_palindrome,
    "keywords:keyword_specific_position": _keywords_specific_position,
    "keywords:start_end": _keywords_start_end,
}


class IFMultiConstraintsVerifier:
    """Verifier for IF_multi_constraints_upto5 dataset (multi-instruction)."""

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def _parse_ground_truth(self, ground_truth: str | dict | list) -> tuple[list[str], list[dict]]:
        if isinstance(ground_truth, (list, dict)):
            gt = ground_truth
        else:
            try:
                gt = ast.literal_eval(ground_truth)
            except Exception:
                try:
                    gt = json.loads(ground_truth)
                except Exception:
                    return [], []

        if not gt:
            return [], []

        entry = gt[0] if isinstance(gt, list) else gt
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:
                return [], []

        instruction_ids = entry.get("instruction_id", [])
        kwargs_list = entry.get("kwargs", [])
        if not isinstance(instruction_ids, list) or not isinstance(kwargs_list, list):
            return [], []
        return instruction_ids, kwargs_list

    def verify(self, response: str, ground_truth: str | dict | list) -> float:
        instruction_ids, kwargs_list = self._parse_ground_truth(ground_truth)
        if not instruction_ids:
            return 0.0

        answer = _remove_thinking_section(response)
        if not answer:
            return 0.0

        rewards = []
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
            func = INSTRUCTION_FUNCTIONS.get(instruction_id)
            if func is None:
                rewards.append(0.0)
                continue
            if kwargs is None:
                kwargs = {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                ok = func(answer, **kwargs) if kwargs else func(answer)
            except Exception:
                ok = False
            rewards.append(1.0 if ok else 0.0)
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)
        
    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        ground_truth = problem.get("ground_truth", "")
        return [self.verify(c, ground_truth) for c in completions]

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        scores = []
        durations = []
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p.get("ground_truth", ""))
            dur = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur)
        return scores, durations

    async def verify_batch_with_timing(
        self, problems: list[dict], completions: list[str]
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        scores = []
        durations = []
        timing_spans = []
        offset = 0.0
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p.get("ground_truth", ""))
            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset, dur_ms))
            offset += dur_ms
        return scores, durations, timing_spans
