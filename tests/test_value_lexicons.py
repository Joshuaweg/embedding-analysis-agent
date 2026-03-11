"""Tests for value_lexicons.py — MFT and Schwartz value token lists."""
import pytest

from value_lexicons import (
    MORAL_FOUNDATIONS,
    SCHWARTZ_VALUES,
    get_all_tokens,
    get_mft_pole_tokens,
    get_schwartz_tokens,
)


def test_moral_foundations_has_five_categories():
    assert len(MORAL_FOUNDATIONS) == 5


def test_each_mft_category_has_both_poles():
    for name, foundation in MORAL_FOUNDATIONS.items():
        assert "positive" in foundation, f"{name} missing 'positive'"
        assert "negative" in foundation, f"{name} missing 'negative'"


def test_schwartz_has_ten_categories():
    assert len(SCHWARTZ_VALUES) == 10


def test_all_tokens_are_strings():
    for foundation in MORAL_FOUNDATIONS.values():
        for pole_tokens in foundation.values():
            for token in pole_tokens:
                assert isinstance(token, str), f"Non-string token: {token!r}"
    for category_tokens in SCHWARTZ_VALUES.values():
        for token in category_tokens:
            assert isinstance(token, str), f"Non-string token: {token!r}"


def test_no_empty_token_lists():
    for name, foundation in MORAL_FOUNDATIONS.items():
        for pole, tokens in foundation.items():
            assert len(tokens) > 0, f"{name}.{pole} is empty"
    for name, tokens in SCHWARTZ_VALUES.items():
        assert len(tokens) > 0, f"Schwartz {name} is empty"


def test_get_all_tokens_returns_list():
    result = get_all_tokens()
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_mft_pole_tokens_care_positive_contains_care():
    tokens = get_mft_pole_tokens("care_harm", "positive")
    assert "care" in tokens or " care" in tokens


def test_space_prefix_variants_exist():
    """For a sample of tokens, both bare and ' token' form should be present."""
    sample_bare = ["care", "harm", "fair", "loyal", "pure", "power", "freedom"]
    all_tokens = get_all_tokens("both")
    for bare in sample_bare:
        assert bare in all_tokens, f"Bare token '{bare}' missing"
        assert f" {bare}" in all_tokens, f"Space-prefixed ' {bare}' missing"
