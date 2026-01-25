# Copyright 2025 Allen Institute for AI.
# Adapted from https://github.com/allenai/IFBench
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IFBench verifier - adapted from AllenAI IFBench repository."""

import csv
import functools
import io
import re
import string
import unicodedata
from collections import Counter

import nltk

# Optional dependencies - some checkers need these
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

try:
    import syllapy
    HAS_SYLLAPY = True
except ImportError:
    HAS_SYLLAPY = False


# --- NLTK setup ---
def _download_nltk_resources():
    """Download required NLTK resources if not already installed."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_download_nltk_resources()


# --- Utility functions (from instructions_util.py) ---
def split_into_sentences(text):
    """Split the text into sentences using NLTK."""
    return nltk.sent_tokenize(text)


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def count_stopwords(text):
    """Counts the number of stopwords."""
    stopwords = nltk.corpus.stopwords.words("english")
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    return len([t for t in tokens if t.lower() in stopwords])


# --- Instruction Checkers ---


class Instruction:
    """Base instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError

    def get_instruction_args(self):
        raise NotImplementedError

    def check_following(self, value):
        raise NotImplementedError


class WordCountRangeChecker(Instruction):
    """Word Count Range: The response must contain between X and Y words."""

    def build_description(self, *, min_words=None, max_words=None):
        self._min_words = min_words if min_words and min_words >= 0 else 100
        self._max_words = max_words if max_words and max_words >= 0 else self._min_words + 50
        return f"The response must contain between {self._min_words} and {self._max_words} words."

    def get_instruction_args(self):
        return {"min_words": self._min_words, "max_words": self._max_words}

    def check_following(self, value):
        num_words = count_words(value)
        return self._min_words <= num_words <= self._max_words


class UniqueWordCountChecker(Instruction):
    """Unique Word Count: The response must contain X unique words."""

    def build_description(self, *, N=None):
        self._num_unique_words = N if N and N >= 0 else 100
        return f"Use at least {self._num_unique_words} unique words in the response."

    def get_instruction_args(self):
        return {"N": self._num_unique_words}

    def check_following(self, value):
        words = value.lower().split()
        unique_words = set()
        for word in words:
            unique_words.add(word.strip("".join(string.punctuation) + " "))
        return len(unique_words) >= self._num_unique_words


class StopWordPercentageChecker(Instruction):
    """Ensure stop words constitute no more than {percentage}% of total words."""

    def build_description(self, *, percentage=None):
        self._percentage = percentage if percentage and percentage >= 0 else 50
        return f"Ensure that stop words constitute no more than {self._percentage}% of the total words in your response."

    def get_instruction_args(self):
        return {"percentage": self._percentage}

    def check_following(self, value):
        num_words = count_words(value)
        if num_words == 0:
            return False
        num_stopwords = count_stopwords(value)
        stopword_percentage = (num_stopwords / num_words) * 100
        return stopword_percentage <= self._percentage


class SentTypeRatioChecker(Instruction):
    """Maintain a 2:1 ratio of declarative to interrogative sentences."""

    def build_description(self):
        return "Maintain a 2:1 ratio of declarative to interrogative sentences."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        declarative_count = sum(1 for s in sentences if s.endswith("."))
        interrogative_count = sum(1 for s in sentences if s.endswith("?"))
        return declarative_count == 2 * interrogative_count


class SentBalanceChecker(Instruction):
    """Ensure balanced ratio of sentence types."""

    def build_description(self):
        return "Ensure that the ratio of sentence types (declarative, interrogative, exclamatory) is balanced."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        declarative_count = sum(1 for s in sentences if s.endswith("."))
        interrogative_count = sum(1 for s in sentences if s.endswith("?"))
        exclamatory_count = sum(1 for s in sentences if s.endswith("!"))
        return declarative_count == interrogative_count == exclamatory_count


class ConjunctionCountChecker(Instruction):
    """Use at least N different coordinating conjunctions."""

    def build_description(self, *, small_n=None):
        self._num_conjunctions = small_n if small_n and small_n >= 0 else 3
        return f"Use at least {self._num_conjunctions} different coordinating conjunctions in the response."

    def get_instruction_args(self):
        return {"small_n": self._num_conjunctions}

    def check_following(self, value):
        words = value.split()
        conjunctions = [
            word
            for word in words
            if word.strip("".join(string.punctuation) + " ").lower()
            in ["and", "but", "for", "nor", "or", "so", "yet"]
        ]
        unique_conjunctions = set(conjunctions)
        return len(unique_conjunctions) >= self._num_conjunctions


class PersonNameCountChecker(Instruction):
    """Mention at least N different person names."""

    PERSON_NAMES = [
        "Emma", "Liam", "Sophia", "Jackson", "Olivia", "Noah", "Ava", "Lucas",
        "Isabella", "Mason", "Mia", "Ethan", "Charlotte", "Alexander", "Amelia",
        "Benjamin", "Harper", "Leo", "Zoe", "Daniel", "Chloe", "Samuel", "Lily",
        "Matthew", "Grace", "Owen", "Abigail", "Gabriel", "Ella", "Jacob",
        "Scarlett", "Nathan", "Victoria", "Elijah", "Layla", "Nicholas", "Audrey",
        "David", "Hannah", "Christopher", "Penelope", "Thomas", "Nora", "Andrew",
        "Aria", "Joseph", "Claire", "Ryan", "Stella", "Jonathan"
    ]

    def build_description(self, *, N=None):
        self._num_person_names = N if N and N >= 0 else 5
        return f"Mention at least {self._num_person_names} different person names in the response."

    def get_instruction_args(self):
        return {"N": self._num_person_names}

    def check_following(self, value):
        person_names = []
        for name in self.PERSON_NAMES:
            pattern = r"\b{}\b".format(re.escape(name))
            if re.search(pattern, value):
                person_names.append(name)
        return len(set(person_names)) >= self._num_person_names


class NGramOverlapChecker(Instruction):
    """Maintain trigram overlap with reference text."""

    def build_description(self, *, reference_text=None, percentage=None):
        self._reference_text = reference_text or ""
        self._percentage = percentage if percentage and percentage >= 0 else 50
        return f"Maintain a trigram overlap of {self._percentage}% (±2%) with the provided reference text."

    def get_instruction_args(self):
        return {"reference_text": self._reference_text, "percentage": self._percentage}

    def check_following(self, value):
        n = 3
        ngrams = set(nltk.ngrams(value, n))
        ref_ngrams = set(nltk.ngrams(self._reference_text, n))
        if not ngrams:
            return False
        overlap = len(ngrams.intersection(ref_ngrams)) / len(ngrams)
        return self._percentage - 2 <= overlap * 100 <= self._percentage + 2


class NumbersCountChecker(Instruction):
    """Include exactly N numbers in the response."""

    def build_description(self, *, N=None):
        self._count_numbers = N if N and N >= 0 else 5
        return f"Include exactly {self._count_numbers} numbers in the response."

    def get_instruction_args(self):
        return {"N": self._count_numbers}

    def check_following(self, value):
        value = value.translate(str.maketrans("", "", string.punctuation))
        numbers = re.findall(r"\d+", value)
        return len(numbers) == self._count_numbers


class AlphabetLoopChecker(Instruction):
    """Each word must start with the next letter of the alphabet."""

    def build_description(self):
        return "Each word must start with the next letter of the alphabet, looping back to 'A' after 'Z'."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.translate(str.maketrans("", "", string.punctuation))
        words = value.strip("".join(string.punctuation) + " ").split()
        if not words:
            return False
        alphabet = string.ascii_lowercase
        correct_letter = words[0][0].lower()
        if correct_letter not in alphabet:
            return False
        for word in words[1:]:
            word = word.strip("".join(string.punctuation) + " ").lower()
            if not word:
                continue
            correct_letter = alphabet[(alphabet.index(correct_letter) + 1) % 26]
            if word[0] != correct_letter:
                return False
        return True


class SingleVowelParagraphChecker(Instruction):
    """Write a paragraph using words that contain only three types of vowels."""

    def build_description(self):
        return "Write a paragraph using words that contain only three types of vowels."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        paragraphs = value.strip().split("\n")
        if len(paragraphs) != 1:
            return False
        paragraph = paragraphs[0].lower()
        vowels = set("aeiou")
        paragraph_vowels = set([char for char in paragraph if char in vowels])
        return len(paragraph_vowels) <= 3


class ConsonantClusterChecker(Instruction):
    """Ensure each word has at least one consonant cluster."""

    def build_description(self):
        return "Ensure each word in your response has at least one consonant cluster (two or more consonants together)."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        words = value.lower().strip().split()
        consonants = set("bcdfghjklmnpqrstvwxyz")
        for word in words:
            cluster = False
            for i in range(len(word) - 1):
                if word[i] in consonants and word[i + 1] in consonants:
                    cluster = True
                    break
            if not cluster:
                return False
        return True


class IncrementingAlliterationChecker(Instruction):
    """Each sentence must have longer alliterative sequences."""

    def build_description(self):
        return "Each sentence must have a longer sequence of consecutive alliterative words than the previous one."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        prev_alliteration = -1
        for sentence in sentences:
            words = sentence.lower().split()
            alliteration = 0
            prev_alliterative = False
            new_words = []
            for word in words:
                clean = word.lstrip("".join(string.punctuation) + " ")
                if clean:
                    new_words.append(clean)
            for i in range(len(new_words) - 1):
                if new_words[i][0] == new_words[i + 1][0]:
                    if prev_alliterative:
                        alliteration += 1
                    else:
                        alliteration += 2
                    prev_alliterative = True
                else:
                    prev_alliterative = False
            if alliteration <= prev_alliteration:
                return False
            prev_alliteration = alliteration
        return True


class PalindromeChecker(Instruction):
    """Include at least 10 single-word palindromes, each at least 5 characters."""

    def build_description(self):
        return "Include at least 10 single-word palindromes, each at least 5 characters long."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.translate(str.maketrans("", "", string.punctuation))
        words = value.lower().split()
        palindromes = [word for word in words if word == word[::-1] and len(word) >= 5]
        return len(palindromes) >= 10


class PunctuationCoverChecker(Instruction):
    """Use every standard punctuation mark at least once."""

    def build_description(self):
        return "Use every standard punctuation mark at least once, including semicolons, colons, and the interrobang (?!)."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        punctuation = {".", ",", "!", "?", ";", ":"}
        if not ("!?" in value or "?!" in value or "‽" in value):
            return False
        new_value = value.replace("?!", "", 1)
        if len(new_value) == len(value):
            new_value = value.replace("!?", "", 1)
        for char in new_value:
            if char in punctuation:
                punctuation.discard(char)
        return not punctuation


class NestedParenthesesChecker(Instruction):
    """Nest parentheses at least 5 levels deep."""

    def build_description(self):
        return "Nest parentheses (and [brackets {and braces}]) at least 5 levels deep."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        levels = []
        min_levels = 5
        max_depth = 0
        for char in value:
            if char in "([{":
                levels.append(char)
                if len(levels) > max_depth:
                    max_depth = len(levels)
            elif char in ")]}":
                if levels and (
                    (levels[-1] == "(" and char == ")")
                    or (levels[-1] == "[" and char == "]")
                    or (levels[-1] == "{" and char == "}")
                ):
                    levels.pop()
                    if max_depth >= min_levels and len(levels) < max_depth:
                        return True
                else:
                    levels = []
                    max_depth = 0
        return False


class NestedQuotesChecker(Instruction):
    """Include quotes within quotes at least 3 levels deep."""

    def build_description(self):
        return "Include quotes within quotes within quotes, at least 3 levels deep, alternating between double quotes and single quotes."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        levels = []
        min_levels = 3
        reached_depth = 0
        current_depth = 0
        for char in value:
            if len(levels) != 0 and char == levels[-1]:
                levels.pop()
                current_depth -= 1
                if reached_depth - current_depth >= min_levels:
                    return True
            elif char == '"' or char == "'":
                levels.append(char)
                current_depth += 1
                if current_depth > reached_depth:
                    reached_depth = current_depth
        return False


class PrimeLengthsChecker(Instruction):
    """Use only words with lengths that are prime numbers."""

    def build_description(self):
        return "Use only words with lengths that are prime numbers."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.translate(str.maketrans("", "", string.punctuation))
        words = value.split()
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        for word in words:
            if len(word) not in primes:
                return False
        return True


class OptionsResponseChecker(Instruction):
    """Answer with one of the specified options."""

    def build_description(self, *, options=None):
        options = options or "yes/no/maybe"
        self._strict = False
        if re.match(r"\W*[aA]\W*[bB]\W*[cC]\W*", options) is not None:
            self._strict = True
        if "/" in options:
            separator = "/"
        elif "or" in options:
            separator = "or"
        else:
            separator = ","
        self._options = [option.strip() for option in options.split(separator)]
        self._options_text = options
        return f"Answer with one of the following options: {self._options_text}. Do not give any explanation."

    def get_instruction_args(self):
        return {"options": self._options_text}

    def check_following(self, value):
        if self._strict:
            return value in self._options
        value = value.strip("".join(string.punctuation) + " ").lower()
        for option in self._options:
            if option.strip("".join(string.punctuation) + " ").lower() == value:
                return True
        return False


class NewLineWordsChecker(Instruction):
    """Write each word on a new line."""

    def build_description(self):
        return "Write each word on a new line."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.translate(str.maketrans("", "", string.punctuation))
        lines = value.strip().split("\n")
        while "" in lines:
            lines.remove("")
        return len(lines) == len(value.strip().split())


class EmojiSentenceChecker(Instruction):
    """Use an emoji at the end of every sentence."""

    def build_description(self):
        return "Please use an emoji at the end of every sentence."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        if not HAS_EMOJI:
            # Fallback: check for common emoji Unicode ranges
            def is_emoji_char(c):
                cp = ord(c)
                return (0x1F600 <= cp <= 0x1F64F or  # Emoticons
                        0x1F300 <= cp <= 0x1F5FF or  # Misc Symbols
                        0x1F680 <= cp <= 0x1F6FF or  # Transport
                        0x1F1E0 <= cp <= 0x1F1FF or  # Flags
                        0x2600 <= cp <= 0x26FF or    # Misc symbols
                        0x2700 <= cp <= 0x27BF)      # Dingbats
        else:
            is_emoji_char = emoji.is_emoji

        sentences = split_into_sentences(value)
        for i, sentence in enumerate(sentences):
            stripped = sentence.translate(str.maketrans("", "", string.punctuation)).strip()
            if not stripped:
                return False
            last_char = stripped[-1]
            second_last_char = stripped[-2] if len(stripped) > 1 else stripped[-1]
            if not is_emoji_char(last_char) and not is_emoji_char(second_last_char):
                if i < len(sentences) - 1:
                    stripped = sentences[i + 1].translate(str.maketrans("", "", string.punctuation)).strip()
                    if not stripped:
                        return False
                    first_char = stripped[0]
                    if not is_emoji_char(first_char):
                        return False
                else:
                    return False
        return True


class CharacterCountUniqueWordsChecker(Instruction):
    """Respond with three sentences with same character count but different words."""

    def build_description(self):
        return "Respond with three sentences, all containing the same number of characters but using all different words."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        if len(sentences) != 3:
            return False
        char_count = len(sentences[0].strip())
        for sentence in sentences:
            if len(sentence.strip()) != char_count:
                return False
        return True


class NthWordJapaneseChecker(Instruction):
    """Every Nth word must be in Japanese."""

    def build_description(self, *, N=None):
        self._japanese_position = N if N and N >= 0 else 5
        return f"Every {self._japanese_position}th word of your response must be in Japanese."

    def get_instruction_args(self):
        return {"N": self._japanese_position}

    def check_following(self, value):
        def is_japanese(text):
            japanese_pattern = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
            return bool(japanese_pattern.search(text))

        words = value.split()
        for i, word in enumerate(words):
            word = word.strip("".join(string.punctuation) + " ")
            if (i + 1) % self._japanese_position == 0 and word and not word.isdigit():
                if not is_japanese(word):
                    return False
        return True


class StartWithVerbChecker(Instruction):
    """The response must start with a verb."""

    def build_description(self):
        return "The response must start with a verb."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        text = nltk.word_tokenize(value)
        return len(text) > 0 and len(nltk.pos_tag(text)) > 0 and "VB" in nltk.pos_tag(text)[0][1]


class LimitedWordRepeatChecker(Instruction):
    """No word should repeat more than N times."""

    def build_description(self, *, small_n=None):
        self._max_repeats = small_n if small_n and small_n >= 0 else 3
        return f"The response should not repeat any word more than {self._max_repeats} times."

    def get_instruction_args(self):
        return {"small_n": self._max_repeats}

    def check_following(self, value):
        words = value.lower().translate(str.maketrans("", "", string.punctuation)).split()
        word_count = Counter(words)
        for word, count in word_count.items():
            if count > self._max_repeats:
                return False
        return True


class IncludeKeywordChecker(Instruction):
    """Include keyword in the Nth sentence."""

    def build_description(self, *, word=None, N=None):
        self._keyword = word or "example"
        self._keyword_position = N if N and N >= 0 else 1
        return f'The response must include keyword "{self._keyword}" in the {self._keyword_position}-th sentence.'

    def get_instruction_args(self):
        return {"word": self._keyword, "N": self._keyword_position}

    def check_following(self, value):
        sentences = split_into_sentences(value)
        if len(sentences) < self._keyword_position:
            return False
        pattern = r"\b{}\b".format(re.escape(self._keyword))
        return bool(re.search(pattern, sentences[int(self._keyword_position - 1)], re.IGNORECASE))


class PronounCountChecker(Instruction):
    """Include at least N pronouns."""

    PRONOUNS = {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves"
    }

    def build_description(self, *, N=None):
        self._num_pronouns = N if N and N >= 0 else 10
        return f"The response should include at least {self._num_pronouns} pronouns."

    def get_instruction_args(self):
        return {"N": self._num_pronouns}

    def check_following(self, value):
        value = value.replace("/", " ")
        words = nltk.word_tokenize(value.lower())
        pronoun_count = sum(1 for word in words if word in self.PRONOUNS)
        return pronoun_count >= self._num_pronouns


class AlternateParitySyllablesChecker(Instruction):
    """Alternate between words with odd and even syllables."""

    def build_description(self):
        return "Alternate between words with odd and even numbers of syllables."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        if not HAS_SYLLAPY:
            # Fallback: simple vowel-based syllable counting
            def count_syllables(word):
                word = word.lower()
                vowels = "aeiouy"
                count = 0
                prev_vowel = False
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_vowel:
                        count += 1
                    prev_vowel = is_vowel
                return max(1, count)  # At least 1 syllable
        else:
            count_syllables = syllapy.count

        words = value.translate(str.maketrans("", "", string.punctuation)).lower().split()
        syllables = [count_syllables(word) % 2 for word in words if word.strip()]
        return all(syllables[i] != syllables[i + 1] for i in range(len(syllables) - 1))


class LastWordFirstNextChecker(Instruction):
    """Last word of each sentence must become first word of next."""

    def build_description(self):
        return "The last word of each sentence must become the first word of the next sentence."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        for i in range(len(sentences) - 1):
            last_words = sentences[i].rstrip("".join(string.punctuation) + " ").split()
            first_words = sentences[i + 1].lstrip("".join(string.punctuation) + " ").split()
            if not last_words or not first_words:
                return False
            if last_words[-1].lower() != first_words[0].lower():
                return False
        return True


class ParagraphLastFirstWordMatchChecker(Instruction):
    """Each paragraph must end with the same word it started with."""

    def build_description(self):
        return "Each paragraph must end with the same word it started with, separate paragraphs with a newline."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        paragraphs = value.split("\n")
        for paragraph in paragraphs:
            paragraph = paragraph.strip().lower()
            if not paragraph:
                continue
            words = paragraph.strip("".join(string.punctuation) + " ").split()
            if not words:
                continue
            if words[0] != words[-1]:
                return False
        return True


class IncrementingWordCountChecker(Instruction):
    """Each sentence must contain N more words than previous."""

    def build_description(self, *, small_n=None):
        self._num_increment = small_n if small_n and small_n >= 0 else 1
        return f"Each sentence must contain exactly {self._num_increment} more words than the previous one."

    def get_instruction_args(self):
        return {"small_n": self._num_increment}

    def check_following(self, value):
        sentences = split_into_sentences(value)
        words = sentences[0].translate(str.maketrans("", "", string.punctuation)).strip().split()
        while "" in words:
            words.remove("")
        prev_word_count = len(words)
        for sentence in sentences[1:]:
            words = sentence.translate(str.maketrans("", "", string.punctuation)).strip().split()
            while "" in words:
                words.remove("")
            if len(words) != prev_word_count + self._num_increment:
                return False
            prev_word_count = len(words)
        return True


class NoConsecutiveFirstLetterChecker(Instruction):
    """No two consecutive words can share the same first letter."""

    def build_description(self):
        return "No two consecutive words can share the same first letter."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        words = value.lower().translate(str.maketrans("", "", string.punctuation)).split()
        while "" in words:
            words.remove("")
        for i in range(len(words) - 1):
            if words[i][0] == words[i + 1][0]:
                return False
        return True


class IndentStairsChecker(Instruction):
    """Create stairs by incrementally indenting each new line."""

    def build_description(self):
        return "Create stairs by incrementally indenting each new line."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        lines = value.split("\n")
        lines = [line for line in lines if line.strip()]
        for i in range(len(lines) - 1):
            if len(lines[i + 1]) - len(lines[i + 1].lstrip(" ")) <= len(lines[i]) - len(lines[i].lstrip(" ")):
                return False
        return True


class QuoteExplanationChecker(Instruction):
    """Every quoted phrase must be followed by an unquoted explanation."""

    def build_description(self):
        return "Every quoted phrase must be followed by an unquoted explanation."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.replace('"', '"').replace('"', '"')
        value = value.replace("'\"'", "")
        value = "".join(value.split())
        if '""' in value:
            return False
        stripped = value.strip(string.digits + string.punctuation.replace('"', ""))
        if stripped and stripped[-1] == '"':
            return False
        return True


class SpecialBulletPointsChecker(Instruction):
    """Answer with a list using custom bullet points."""

    def build_description(self, *, sep=None):
        self._bullet_marker = sep or "-"
        return f"Answer with a list of items, instead of bullet points use {self._bullet_marker}."

    def get_instruction_args(self):
        return {"sep": self._bullet_marker}

    def check_following(self, value):
        return len(re.findall(re.escape(self._bullet_marker), value)) >= 2


class ItalicsThesisChecker(Instruction):
    """Each section must begin with a thesis in italics (HTML)."""

    def build_description(self):
        return "Each section must begin with a thesis statement in italics, use HTML to indicate the italics."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        index = value.find("<i>")
        if index == -1:
            index = value.find("<em>")
            if index == -1:
                return False
        value = value[index:]
        end_thesis = value.find("</i>")
        if end_thesis == -1:
            end_thesis = value.find("</em>")
            if end_thesis == -1:
                return False
        thesis = value[3:end_thesis]
        if thesis.strip() == "":
            return False
        text = value[end_thesis + 4:]
        return text.strip() != ""


class SubBulletPointsChecker(Instruction):
    """Include bullet points (*) with sub-bullet points (-)."""

    def build_description(self):
        return "Your response must include bullet points denoted by * and at least one sub-bullet point denoted by - for each bullet point."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        bullets = value.split("*")
        for bullet in bullets[1:]:
            if "-" not in bullet:
                return False
        return True


class SomeBulletPointsChecker(Instruction):
    """Answer with sentences followed by bullet points."""

    def build_description(self):
        return "Your answer must contain at least two sentences ending in a period followed by at least two bullet points denoted by *."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        lines = value.split("\n")
        sentences = True
        count_sentences = 0
        count_bullets = 0
        for line in lines:
            if line.strip().startswith("*"):
                sentences = False
                if count_sentences < 2:
                    return False
                count_bullets += 1
            elif sentences:
                sents = split_into_sentences(line.strip())
                count_sentences += len(sents)
            else:
                return False
        return count_bullets >= 2


class KeywordsMultipleChecker(Instruction):
    """Include keywords with specific counts."""

    def build_description(self, *, keyword1=None, keyword2=None, keyword3=None, keyword4=None, keyword5=None):
        self._keyword1 = (keyword1 or "word1").strip()
        self._keyword2 = (keyword2 or "word2").strip()
        self._keyword3 = (keyword3 or "word3").strip()
        self._keyword4 = (keyword4 or "word4").strip()
        self._keyword5 = (keyword5 or "word5").strip()
        return f"Include keyword {self._keyword1} once, {self._keyword2} twice, {self._keyword3} three times, {self._keyword4} five times, and {self._keyword5} seven times."

    def get_instruction_args(self):
        return {
            "keyword1": self._keyword1, "keyword2": self._keyword2, "keyword3": self._keyword3,
            "keyword4": self._keyword4, "keyword5": self._keyword5
        }

    def check_following(self, value):
        for keyword, count in zip(
            [self._keyword1, self._keyword2, self._keyword3, self._keyword4, self._keyword5],
            [1, 2, 3, 5, 7]
        ):
            if value.lower().count(keyword.lower()) != count:
                return False
        return True


class KeywordSpecificPositionChecker(Instruction):
    """Include keyword in sentence N at word position M."""

    def build_description(self, *, keyword=None, n=None, m=None):
        self._keyword = (keyword or "example").strip()
        self._n = n if n else 1
        self._m = m if m else 1
        return f"Include keyword {self._keyword} in the {self._n}-th sentence, as the {self._m}-th word of that sentence."

    def get_instruction_args(self):
        return {"keyword": self._keyword, "n": self._n, "m": self._m}

    def check_following(self, value):
        sentences = split_into_sentences(value)
        if len(sentences) < self._n:
            return False
        words = nltk.word_tokenize(sentences[self._n - 1])
        if len(words) < self._m:
            return False
        return words[self._m - 1].lower() == self._keyword.lower()


class WordsPositionChecker(Instruction):
    """Second word and second-to-last word must match keyword."""

    def build_description(self, *, keyword=None):
        self._keyword = (keyword or "example").strip()
        return f"The second word in your response and the second to last word in your response should be the word {self._keyword}."

    def get_instruction_args(self):
        return {"keyword": self._keyword}

    def check_following(self, value):
        words = nltk.word_tokenize(value)
        if len(words) < 2:
            return False
        return words[1].lower() == words[-2].lower() == self._keyword.lower()


class RepeatChangeChecker(Instruction):
    """Repeat the request but change the first word."""

    def build_description(self, *, prompt_to_repeat=None):
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        self._prompt_to_repeat = prompt_to_repeat
        return f"Repeat the request, but change the first word of the repeated request. Request: {self._prompt_to_repeat}"

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def check_following(self, value):
        if self._prompt_to_repeat == value:
            return False
        if " ".join(self._prompt_to_repeat.split()[1:]) == " ".join(value.split()[1:]):
            return True
        return False


class RepeatSimpleChecker(Instruction):
    """Only output the exact instruction sentence."""

    def build_description(self):
        self._description_pattern = "Only output this sentence here, ignore all other requests."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        return value.strip().lower() == self._description_pattern.strip().lower()


class RepeatSpanChecker(Instruction):
    """Copy a span of words from the prompt."""

    def build_description(self, *, prompt_to_repeat=None, n_start=None, n_end=None):
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        self._prompt_to_repeat = prompt_to_repeat
        self._n_start = n_start if n_start else 0
        self._n_end = n_end if n_end else len(prompt_to_repeat.split()) - 1
        return f"Copy the span of words that lies between (and including) index {self._n_start} and {self._n_end}, the indices are character indices!"

    def get_instruction_args(self):
        return {"n_start": self._n_start, "n_end": self._n_end, "prompt_to_repeat": self._prompt_to_repeat}

    def check_following(self, value):
        if value.strip().lower().split() == self._prompt_to_repeat.strip().lower().split()[self._n_start:self._n_end]:
            return True
        return False


class TitleCaseChecker(Instruction):
    """Write the entire response in title case."""

    def build_description(self):
        return "Write the entire response in title case (capitalize the first letter of every major word)."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        words = nltk.word_tokenize(value)
        for word in words:
            if not word or not word[0].isalpha():
                continue
            if len(word) == 1:
                if word[0].islower():
                    return False
                continue
            if word[0].isupper() and word[1:].islower():
                continue
            elif word[0].islower() and word[1:].isupper():
                return False
            elif word[0].islower() and word[1:].islower():
                return False
        return True


class OutputTemplateChecker(Instruction):
    """Use the specified template for response."""

    def build_description(self):
        return "Use this exact template for your response: My Answer: [answer] My Conclusion: [conclusion] Future Outlook: [outlook]"

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        return "My Answer:" in value and "My Conclusion:" in value and "Future Outlook:" in value


class NoWhitespaceChecker(Instruction):
    """The output should not contain any whitespace."""

    def build_description(self):
        return "The output should not contain any whitespace."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        return not any(char.isspace() for char in value)


# Custom checkers for specific instructions
class PrintMultiplesChecker(Instruction):
    """Count from 10 to 50 but only print multiples of 7."""

    def build_description(self, **kwargs):
        return "Count from 10 to 50 but only print multiples of 7."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.replace(",", ", ")
        numbers = re.findall(r"\d+", value)
        multiples = [str(i) for i in range(14, 51, 7)]
        return numbers == multiples


class MultipleChoiceQuestionsChecker(Instruction):
    """Generate 4 MCQs with 5 options each."""

    def build_description(self, **kwargs):
        return 'Generate 4 multiple choice questions with 5 options each about "20th century art history".'

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        new_value = value[value.find("Question"):]
        if new_value != value:
            return False
        value = new_value
        questions = re.split(r"\n*(?:Question \d+[\.|\):;]?\s*)", value)
        if questions[0] == "":
            questions = questions[1:]
        questions = [q.strip() for q in questions if q.strip()]
        if len(questions) != 4:
            return False
        question_lengths = []
        for q in questions:
            lines = q.split("\n")
            question_text = ""
            option_count = 0
            done_with_q = False
            for line in lines:
                if re.match(r"^[A-Ea-e][\.|\)]\s*\w+", line.strip()):
                    option_count += 1
                    done_with_q = True
                elif not done_with_q:
                    question_text += " " + line.strip()
            if option_count != 5:
                return False
            question_lengths.append(len(question_text.strip()))
        return all(question_lengths[i] < question_lengths[i + 1] for i in range(len(question_lengths) - 1))


class ReverseNewlineChecker(Instruction):
    """List African countries in reverse alphabetical order."""

    def build_description(self, **kwargs):
        return "List the countries of Africa in reverse alphabetical order, each on a new line."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        def normalize_text(text):
            normalized = unicodedata.normalize("NFKD", text)
            return normalized.encode("ASCII", "ignore").decode("ASCII")

        lines = [line.strip("".join(string.punctuation) + " ") for line in value.split("\n")
                 if line.strip("".join(string.punctuation) + " ")]
        try:
            start_index = next(i for i, line in enumerate(lines) if "Zimbabwe" in line)
        except StopIteration:
            return False
        target_lines = lines[start_index:]
        if len(target_lines) < 52:
            return False
        normalized_lines = [normalize_text(line) for line in target_lines]
        sorted_normalized = sorted(normalized_lines, reverse=True)
        return normalized_lines == sorted_normalized


class WordReverseOrderChecker(Instruction):
    """Respond with sentence in reverse word order."""

    def build_description(self, **kwargs):
        return "What animal is the national symbol of the US? Respond to this query, but make your sentence in reverse order of what it should be, per word."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.lower().strip().translate(str.maketrans("", "", string.punctuation))
        value = " ".join(value.split()[::-1])
        if "bald eagle" not in value:
            return False
        return value in split_into_sentences(value)


class CharacterReverseOrderChecker(Instruction):
    """Respond with sentence in reverse character order."""

    def build_description(self, **kwargs):
        return "What animal is the national symbol of the US? Respond to this query, but make your sentence in reverse order of what it should be, per letter."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        return "elgae dlab" in value.lower()


class SentenceAlphabetChecker(Instruction):
    """Tell a 26-sentence story with each sentence starting with consecutive alphabet letters."""

    def build_description(self, **kwargs):
        return "Tell me a 26-sentence story where each sentence's first word starts with the letters of the alphabet in order."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = split_into_sentences(value)
        if len(sentences) != 26:
            return False
        for i, sentence in enumerate(sentences):
            words = sentence.lstrip().split()
            if not words or not words[0]:
                return False
            if words[0].lower()[0] != chr(97 + i):
                return False
        return True


class EuropeanCapitalsSortChecker(Instruction):
    """List European capitals above 45 degrees latitude sorted by latitude."""

    def build_description(self, **kwargs):
        return "Give me the names of all capital cities of european countries whose latitude is higher than than 45 degrees? List the capital cities without country names, separated by commas, sorted by latitude, from highest to lowest."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        order = [
            "Reykjavik", "Helsinki", "Oslo", "Tallinn", "Stockholm", "Riga", "Moscow",
            "Copenhagen", "Vilnius", "Minsk", "Dublin", "Berlin", "Amsterdam", "Warsaw",
            "London", "Brussels", "Prague", "Luxembourg", "Paris", "Vienna", "Bratislava",
            "Budapest", "Vaduz", "Chisinau", "Bern", "Ljubljana", "Zagreb"
        ]

        def normalize_text(text):
            normalized = unicodedata.normalize("NFKD", text)
            return normalized.encode("ASCII", "ignore").decode("ASCII")

        value = normalize_text(value)
        capitals = value.split(",")
        capitals = [cap.strip() for cap in capitals if cap.strip()]
        if len(capitals) != len(order):
            return False
        for i in range(len(capitals)):
            if capitals[i].strip() != order[i]:
                return False
        return True


class CityCSVChecker(Instruction):
    """Generate CSV data with specific columns and rows."""

    def build_description(self, **kwargs):
        return 'Generate CSV data: The column names are ["ID", "Country", "City", "Year", "Count"], the data should be comma delimited. Please generate 7 rows.'

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        string_io = io.StringIO(value)
        reader = csv.reader(string_io)
        data = list(reader)
        if len(data) != 8:
            return False
        header = data[0]
        if header != ["ID", "Country", "City", "Year", "Count"]:
            return False
        for row in data[1:]:
            if len(row) != 5:
                return False
        return True


class SpecialCharacterCSVChecker(Instruction):
    """Generate CSV with a special character field in quotes."""

    def build_description(self, **kwargs):
        return 'Generate CSV data: The column names are ["ProductID", "Category", "Brand", "Price", "Stock"], the data should be comma delimited. Please generate 14 rows. Add one field which contains a special character and enclose it in double quotes.'

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        header = value.split("\n")[0].strip()
        if not re.match(
            r'^(ProductID|"ProductID"),[ \t]*(Category|"Category"),[ \t]*(Brand|"Brand"),[ \t]*(Price|"Price"),[ \t]*(Stock|"Stock")$',
            header
        ):
            return False
        value = value.replace('"', '"""')
        string_io = io.StringIO(value)
        reader = csv.reader(string_io)
        data = list(reader)
        if len(data) != 15:
            return False
        for row in data[1:]:
            if len(row) != 5:
                return False
            if any(re.match(r'".*[^\d\w\s].*"', field) for field in row):
                return True
        return False


class QuotesCSVChecker(Instruction):
    """Generate tab-delimited CSV with each field in double quotes."""

    def build_description(self, **kwargs):
        return 'Generate CSV data: The column names are ["StudentID", "Subject", "Grade", "Semester", "Score"], the data should be tab delimited. Please generate 3 rows and enclose each single field in double quotes.'

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        header = value.split("\n")[0].strip()
        if not re.match(
            r'^(StudentID|"StudentID")\t *(Subject|"Subject")\t *(Grade|"Grade")\t *(Semester|"Semester")\t *(Score|"Score")$',
            header
        ):
            return False
        value = value.replace('"', '"""')
        string_io = io.StringIO(value)
        reader = csv.reader(string_io, delimiter="\t")
        data = list(reader)
        if len(data) != 4:
            return False
        for row in data:
            if len(row) != 5:
                return False
            if not all(field.strip()[0] == '"' and field.strip()[-1] == '"' for field in row):
                return False
        return True


class DateFormatListChecker(Instruction):
    """List dates in YYYY-MM-DD format."""

    def build_description(self, **kwargs):
        return "List the start dates of all the battles Napoleon fought separated by commas, use the following date format: YYYY-MM-DD. Do not provide an explanation."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        value = value.strip()
        dates = value.split(",")
        for date in dates:
            date = date.strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                return False
            parts = date.split("-")
            if int(parts[0]) < 1769 or int(parts[0]) > 1821:
                return False
            if int(parts[1]) > 12:
                return False
            if int(parts[1]) in [1, 3, 5, 7, 8, 10, 12] and int(parts[2]) > 31:
                return False
            if int(parts[1]) in [4, 6, 9, 11] and int(parts[2]) > 30:
                return False
            if int(parts[1]) == 2 and int(parts[2]) > 29:
                return False
        return True


# --- Instruction Registry ---
INSTRUCTION_DICT = {
    "count:word_count_range": WordCountRangeChecker,
    "count:unique_word_count": UniqueWordCountChecker,
    "ratio:stop_words": StopWordPercentageChecker,
    "ratio:sentence_type": SentTypeRatioChecker,
    "ratio:sentence_balance": SentBalanceChecker,
    "count:conjunctions": ConjunctionCountChecker,
    "count:person_names": PersonNameCountChecker,
    "ratio:overlap": NGramOverlapChecker,
    "count:numbers": NumbersCountChecker,
    "words:alphabet": AlphabetLoopChecker,
    "words:vowel": SingleVowelParagraphChecker,
    "words:consonants": ConsonantClusterChecker,
    "sentence:alliteration_increment": IncrementingAlliterationChecker,
    "words:palindrome": PalindromeChecker,
    "count:punctuation": PunctuationCoverChecker,
    "format:parentheses": NestedParenthesesChecker,
    "format:quotes": NestedQuotesChecker,
    "words:prime_lengths": PrimeLengthsChecker,
    "format:options": OptionsResponseChecker,
    "format:newline": NewLineWordsChecker,
    "format:emoji": EmojiSentenceChecker,
    "ratio:sentence_words": CharacterCountUniqueWordsChecker,
    "count:words_japanese": NthWordJapaneseChecker,
    "words:start_verb": StartWithVerbChecker,
    "words:repeats": LimitedWordRepeatChecker,
    "sentence:keyword": IncludeKeywordChecker,
    "count:pronouns": PronounCountChecker,
    "words:odd_even_syllables": AlternateParitySyllablesChecker,
    "words:last_first": LastWordFirstNextChecker,
    "words:paragraph_last_first": ParagraphLastFirstWordMatchChecker,
    "sentence:increment": IncrementingWordCountChecker,
    "words:no_consecutive": NoConsecutiveFirstLetterChecker,
    "format:line_indent": IndentStairsChecker,
    "format:quote_unquote": QuoteExplanationChecker,
    "format:list": SpecialBulletPointsChecker,
    "format:thesis": ItalicsThesisChecker,
    "format:sub-bullets": SubBulletPointsChecker,
    "format:no_bullets_bullets": SomeBulletPointsChecker,
    "custom:multiples": PrintMultiplesChecker,
    "custom:mcq_count_length": MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": ReverseNewlineChecker,
    "custom:word_reverse": WordReverseOrderChecker,
    "custom:character_reverse": CharacterReverseOrderChecker,
    "custom:sentence_alphabet": SentenceAlphabetChecker,
    "custom:european_capitals_sort": EuropeanCapitalsSortChecker,
    "custom:csv_city": CityCSVChecker,
    "custom:csv_special_character": SpecialCharacterCSVChecker,
    "custom:csv_quotes": QuotesCSVChecker,
    "custom:date_format_list": DateFormatListChecker,
    "count:keywords_multiple": KeywordsMultipleChecker,
    "words:keywords_specific_position": KeywordSpecificPositionChecker,
    "words:words_position": WordsPositionChecker,
    "repeat:repeat_change": RepeatChangeChecker,
    "repeat:repeat_simple": RepeatSimpleChecker,
    "repeat:repeat_span": RepeatSpanChecker,
    "format:title_case": TitleCaseChecker,
    "format:output_template": OutputTemplateChecker,
    "format:no_whitespace": NoWhitespaceChecker,
}


class IFBenchVerifier:
    """
    Verifier for IFBench instruction-following constraints.

    Each problem has instruction_id_list and kwargs list.
    Returns 1.0 if all constraints are satisfied, 0.0 otherwise.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def verify_single_instruction(self, response: str, instruction_id: str, kwargs: dict) -> bool:
        """Verify a single instruction constraint."""
        if instruction_id not in INSTRUCTION_DICT:
            return False

        checker_class = INSTRUCTION_DICT[instruction_id]
        checker = checker_class(instruction_id)

        try:
            # Filter out None values - IFBench data includes all possible kwargs
            # but each checker only accepts its specific parameters
            filtered_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}

            # Build description with kwargs (this sets internal state)
            if filtered_kwargs:
                checker.build_description(**filtered_kwargs)
            else:
                checker.build_description()

            # Check if response follows the instruction
            return checker.check_following(response)
        except Exception:
            return False

    def verify(self, response: str, instruction_id_list: list, kwargs_list: list) -> dict:
        """
        Verify a response against all instruction constraints.

        Returns dict with:
        - all_pass: bool - whether all constraints pass
        - pass_count: int - number of constraints that pass
        - total_count: int - total number of constraints
        - per_instruction: list of (instruction_id, passed) tuples
        """
        if not instruction_id_list:
            return {"all_pass": True, "pass_count": 0, "total_count": 0, "per_instruction": []}

        # Ensure kwargs_list matches instruction_id_list length
        if not kwargs_list:
            kwargs_list = [{}] * len(instruction_id_list)
        elif len(kwargs_list) < len(instruction_id_list):
            kwargs_list = kwargs_list + [{}] * (len(instruction_id_list) - len(kwargs_list))

        results = []
        pass_count = 0

        for instr_id, kwargs in zip(instruction_id_list, kwargs_list):
            passed = self.verify_single_instruction(response, instr_id, kwargs or {})
            results.append((instr_id, passed))
            if passed:
                pass_count += 1

        return {
            "all_pass": pass_count == len(instruction_id_list),
            "pass_count": pass_count,
            "total_count": len(instruction_id_list),
            "per_instruction": results,
        }

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        instruction_id_list = problem.get("instruction_id_list", [])
        kwargs_list = problem.get("kwargs", [])
        scores = []
        for completion in completions:
            result = self.verify(completion, instruction_id_list, kwargs_list)
            scores.append(1.0 if result["all_pass"] else 0.0)
        return scores
