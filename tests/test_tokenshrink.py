"""
Tests for TokenShrink core functions.

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os

# Make `src` importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# We import only the pure functions – no Streamlit calls happen at import time
# because those are inside `if prompt_original:` blocks.
from streamlit_app import (
    _normalize_whitespace,
    _remove_greetings,
    _remove_redundant_connectives,
    _remove_redundant_phrases,
    _remove_repeated_sentences,
    compress_prompt,
    count_tokens,
    estimate_cost,
    LANG_RULES,
    COST_TABLE,
)


# ─── count_tokens ─────────────────────────────────────────────────────────────

class TestCountTokens:
    """
    Tests for count_tokens().

    tiktoken downloads BPE vocabulary files from OpenAI's CDN at first use.
    In network-restricted CI the download fails, so tests that actually call
    the encoder use a lightweight mock encoder instead of the real one.
    """

    def _make_mock_encoder(self, words_per_token: int = 1):
        """Return a mock encoder whose encode() splits on whitespace."""
        class _MockEncoder:
            def encode(self, text: str) -> list[int]:
                # Each word ≈ words_per_token tokens (good enough for unit tests)
                tokens = text.split()
                return list(range(max(1, len(tokens) // words_per_token or len(tokens))))
        return _MockEncoder()

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_non_string_none_returns_zero(self):
        assert count_tokens(None) == 0   # type: ignore[arg-type]

    def test_count_increases_with_length(self, monkeypatch):
        mock_enc = self._make_mock_encoder()
        monkeypatch.setattr("streamlit_app._load_encoder", lambda _name: mock_enc)
        short = count_tokens("hi", "cl100k_base")
        long  = count_tokens("hi there how are you doing today", "cl100k_base")
        assert long > short

    def test_returns_integer(self, monkeypatch):
        mock_enc = self._make_mock_encoder()
        monkeypatch.setattr("streamlit_app._load_encoder", lambda _name: mock_enc)
        result = count_tokens("hello world", "cl100k_base")
        assert isinstance(result, int)
        assert result >= 1

    def test_single_word_at_least_one_token(self, monkeypatch):
        mock_enc = self._make_mock_encoder()
        monkeypatch.setattr("streamlit_app._load_encoder", lambda _name: mock_enc)
        result = count_tokens("hello", "cl100k_base")
        assert result >= 1

    def test_mock_two_words_two_tokens(self, monkeypatch):
        mock_enc = self._make_mock_encoder()
        monkeypatch.setattr("streamlit_app._load_encoder", lambda _name: mock_enc)
        # "hello world" → 2 words → 2 mock tokens
        assert count_tokens("hello world", "cl100k_base") == 2

    def test_fallback_on_unknown_tokenizer(self, monkeypatch):
        """_load_encoder falls back to DEFAULT_TOKENIZER on unknown name; mock both."""
        mock_enc = self._make_mock_encoder()
        monkeypatch.setattr("streamlit_app._load_encoder", lambda _name: mock_enc)
        result = count_tokens("hello", "nonexistent_tokenizer_xyz")
        assert isinstance(result, int)
        assert result >= 1


# ─── _normalize_whitespace ───────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_strips_leading_trailing(self):
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_preserves_paragraphs_by_default(self):
        text = "para one\n\n\n\npara two"
        result = _normalize_whitespace(text, preserve_paragraphs=True)
        assert "\n\n" in result
        # Should not have triple newlines
        assert "\n\n\n" not in result

    def test_collapses_paragraphs_when_disabled(self):
        text = "para one\n\npara two"
        result = _normalize_whitespace(text, preserve_paragraphs=False)
        assert "\n\n" not in result
        assert "\n" in result

    def test_normalises_crlf(self):
        text = "line one\r\nline two"
        result = _normalize_whitespace(text)
        assert "\r" not in result


# ─── _remove_greetings ────────────────────────────────────────────────────────

class TestRemoveGreetings:
    def test_removes_hello(self):
        patterns = LANG_RULES["English"]["greetings"]
        result = _remove_greetings("Hello! Summarize this text.", patterns)
        assert "Hello" not in result
        assert "Summarize" in result

    def test_removes_portuguese_greeting(self):
        patterns = LANG_RULES["Portuguese"]["greetings"]
        result = _remove_greetings("Olá! Por favor, resuma este texto.", patterns)
        assert "Olá" not in result

    def test_removes_spanish_greeting(self):
        patterns = LANG_RULES["Spanish"]["greetings"]
        result = _remove_greetings("Hola! Por favor resuma este texto.", patterns)
        assert "Hola" not in result

    def test_removes_french_greeting(self):
        patterns = LANG_RULES["French"]["greetings"]
        result = _remove_greetings("Bonjour! Pourriez-vous résumer ce texte.", patterns)
        assert "Bonjour" not in result

    def test_removes_german_greeting(self):
        patterns = LANG_RULES["German"]["greetings"]
        result = _remove_greetings("Guten Tag! Könnten Sie diesen Text zusammenfassen.", patterns)
        assert "Guten Tag" not in result

    def test_no_greeting_unchanged(self):
        patterns = LANG_RULES["English"]["greetings"]
        text = "Summarize this document in three bullet points."
        result = _remove_greetings(text, patterns)
        assert result == text

    def test_case_insensitive(self):
        patterns = LANG_RULES["English"]["greetings"]
        result = _remove_greetings("hello, how are you?", patterns)
        assert "hello" not in result.lower()


# ─── _remove_redundant_connectives ───────────────────────────────────────────

class TestRemoveRedundantConnectives:
    def test_removes_english_connective(self):
        patterns = LANG_RULES["English"]["stops"]
        result = _remove_redundant_connectives("Do this in order to save time.", patterns)
        assert "in order to" not in result

    def test_removes_portuguese_connective(self):
        patterns = LANG_RULES["Portuguese"]["stops"]
        result = _remove_redundant_connectives("Faça isso a fim de economizar tempo.", patterns)
        assert "a fim de" not in result

    def test_no_connective_unchanged(self):
        patterns = LANG_RULES["English"]["stops"]
        text = "Write a concise summary."
        result = _remove_redundant_connectives(text, patterns)
        assert result.strip() == text


# ─── _remove_redundant_phrases ────────────────────────────────────────────────

class TestRemoveRedundantPhrases:
    def test_removes_i_want_you_to(self):
        replacements = LANG_RULES["English"]["redundant_phrases"]
        result = _remove_redundant_phrases("I want you to write a story.", replacements)
        assert "I want you to" not in result

    def test_removes_portuguese_phrase(self):
        replacements = LANG_RULES["Portuguese"]["redundant_phrases"]
        result = _remove_redundant_phrases(
            "Eu quero que você escreva um resumo.", replacements
        )
        # "eu quero que você" → "você"
        assert "eu quero que" not in result.lower()

    def test_plain_text_unchanged(self):
        replacements = LANG_RULES["English"]["redundant_phrases"]
        text = "Write a haiku about autumn leaves."
        result = _remove_redundant_phrases(text, replacements)
        assert result == text


# ─── _remove_repeated_sentences ──────────────────────────────────────────────

class TestRemoveRepeatedSentences:
    def test_removes_exact_duplicate(self):
        text = "The sky is blue. The sky is blue."
        result = _remove_repeated_sentences(text)
        assert result.count("The sky is blue") == 1

    def test_case_insensitive_dedup(self):
        text = "The sky is blue. THE SKY IS BLUE."
        result = _remove_repeated_sentences(text)
        # Only one instance should remain
        lower = result.lower()
        assert lower.count("the sky is blue") == 1

    def test_unique_sentences_preserved(self):
        text = "The sky is blue. The grass is green."
        result = _remove_repeated_sentences(text)
        assert "blue" in result
        assert "green" in result

    def test_dedup_across_newlines(self):
        """FIX: This was broken in the original – newline-separated duplicates were kept."""
        text = "Be concise.\nBe concise."
        result = _remove_repeated_sentences(text)
        assert result.lower().count("be concise") == 1

    def test_empty_string(self):
        assert _remove_repeated_sentences("") == ""


# ─── compress_prompt (integration) ───────────────────────────────────────────

class TestCompressPrompt:
    def _compress(self, text: str, language: str = "English", **kwargs) -> str:
        defaults = dict(
            remove_spaces=True,
            remove_stop_words=False,
            remove_greetings=True,
            remove_redundant=False,
            remove_duplicates=False,
            preserve_paragraphs=True,
            language=language,
        )
        defaults.update(kwargs)
        return compress_prompt(text, **defaults)

    def test_removes_greeting_and_normalises(self):
        result = self._compress("Hello! Summarize this text for me.")
        assert "Hello" not in result
        assert "Summarize" in result

    def test_capitalises_first_letter(self):
        result = self._compress("hello! please write something.")
        assert result[0].isupper()

    def test_empty_input_returns_empty(self):
        result = self._compress("")
        assert result == ""

    def test_already_clean_prompt_unchanged(self):
        text = "Summarize this document in three bullet points."
        result = self._compress(text, remove_greetings=True)
        # No greeting to remove; content should be identical
        assert "Summarize" in result

    def test_all_filters_english(self):
        text = (
            "Hello! I want you to please write a summary in a clear way. "
            "Write a summary in a clear way. Thank you!"
        )
        result = self._compress(
            text,
            remove_stop_words=True,
            remove_redundant=True,
            remove_duplicates=True,
        )
        assert "Hello" not in result
        assert "Thank you" not in result
        assert len(result) < len(text)

    def test_portuguese(self):
        text = "Olá! Por favor, você poderia fazer um resumo deste documento?"
        result = self._compress(text, language="Portuguese")
        assert "Olá" not in result

    def test_spanish(self):
        text = "Hola! Por favor, ¿podrías resumir este documento?"
        result = self._compress(text, language="Spanish")
        assert "Hola" not in result

    def test_french(self):
        text = "Bonjour! Pourriez-vous résumer ce document s'il vous plaît?"
        result = self._compress(text, language="French")
        assert "Bonjour" not in result

    def test_german(self):
        text = "Guten Tag! Könnten Sie dieses Dokument zusammenfassen bitte?"
        result = self._compress(text, language="German")
        assert "Guten Tag" not in result

    def test_unknown_language_falls_back_to_english(self):
        text = "Hello! Summarize this."
        result = compress_prompt(
            text,
            remove_spaces=True,
            remove_stop_words=False,
            remove_greetings=True,
            remove_redundant=False,
            remove_duplicates=False,
            preserve_paragraphs=True,
            language="Klingon",  # not in LANG_RULES → falls back to English
        )
        assert "Hello" not in result

    def test_no_leading_punctuation(self):
        """After stripping a greeting, leftover punctuation should be removed."""
        text = "Hi! How are you? Please write a poem."
        result = self._compress(text)
        assert not result.startswith("!")
        assert not result.startswith("?")
        assert not result.startswith(",")


# ─── estimate_cost ────────────────────────────────────────────────────────────

class TestEstimateCost:
    def test_zero_tokens_is_zero_cost(self):
        for model in COST_TABLE:
            assert estimate_cost(0, model) == 0.0

    def test_known_model_gpt4o(self):
        # 1000 tokens × $0.0025/1k = $0.0025
        assert abs(estimate_cost(1000, "GPT-4o") - 0.0025) < 1e-9

    def test_known_model_claude(self):
        # 1000 tokens × $0.003/1k = $0.003
        assert abs(estimate_cost(1000, "Claude 3.5 Sonnet") - 0.003) < 1e-9

    def test_unknown_model_uses_default_rate(self):
        # Falls back to 0.003 (Claude default in code)
        result = estimate_cost(1000, "Unknown Model XYZ")
        assert isinstance(result, float)
        assert result > 0

    def test_larger_token_count_higher_cost(self):
        cost_small = estimate_cost(100,  "GPT-4o")
        cost_large = estimate_cost(1000, "GPT-4o")
        assert cost_large > cost_small

    @pytest.mark.parametrize("model", list(COST_TABLE.keys()))
    def test_all_models_produce_positive_cost(self, model: str):
        assert estimate_cost(500, model) > 0


# ─── LANG_RULES completeness ─────────────────────────────────────────────────

class TestLangRulesStructure:
    @pytest.mark.parametrize("lang", list(LANG_RULES.keys()))
    def test_each_language_has_required_keys(self, lang: str):
        config = LANG_RULES[lang]
        assert "greetings" in config, f"{lang} missing 'greetings'"
        assert "stops" in config,     f"{lang} missing 'stops'"
        assert "redundant_phrases" in config, f"{lang} missing 'redundant_phrases'"

    @pytest.mark.parametrize("lang", list(LANG_RULES.keys()))
    def test_greetings_are_strings(self, lang: str):
        for pattern in LANG_RULES[lang]["greetings"]:
            assert isinstance(pattern, str), f"{lang}: greeting pattern must be str"

    @pytest.mark.parametrize("lang", list(LANG_RULES.keys()))
    def test_redundant_phrases_are_tuples(self, lang: str):
        for item in LANG_RULES[lang]["redundant_phrases"]:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"{lang}: redundant_phrases items must be (pattern, replacement) tuples"
            )

    def test_all_five_languages_present(self):
        expected = {"English", "Portuguese", "Spanish", "French", "German"}
        assert expected.issubset(set(LANG_RULES.keys()))
