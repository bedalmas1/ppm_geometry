"""Unit tests for src/evaluation/suffix_metrics.py against hand-worked
sequences with analytically known distances."""
import pytest

from evaluation.suffix_metrics import damerau_levenshtein_distance, normalized_dl_similarity


def test_identical_sequences_zero_distance():
    assert damerau_levenshtein_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert normalized_dl_similarity(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_single_substitution():
    assert damerau_levenshtein_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_single_insertion():
    assert damerau_levenshtein_distance(["a", "b"], ["a", "b", "c"]) == 1


def test_single_deletion():
    assert damerau_levenshtein_distance(["a", "b", "c"], ["a", "b"]) == 1


def test_adjacent_transposition_costs_one():
    # Damerau-Levenshtein (unlike plain Levenshtein) counts a swap of two
    # adjacent elements as a single edit, not two.
    assert damerau_levenshtein_distance(["a", "b", "c"], ["a", "c", "b"]) == 1


def test_empty_vs_nonempty_equals_length():
    assert damerau_levenshtein_distance([], ["a", "b", "c"]) == 3
    assert damerau_levenshtein_distance(["a", "b", "c"], []) == 3


def test_both_empty_is_perfectly_similar():
    assert damerau_levenshtein_distance([], []) == 0
    assert normalized_dl_similarity([], []) == pytest.approx(1.0)


def test_normalized_similarity_maximally_dissimilar():
    # completely disjoint tokens, same length -> full substitution needed
    sim = normalized_dl_similarity(["a", "b"], ["x", "y"])
    assert sim == pytest.approx(0.0)


def test_normalized_similarity_matches_hand_calculation():
    # 1 substitution out of max length 3 -> similarity = 1 - 1/3
    sim = normalized_dl_similarity(["a", "b", "c"], ["a", "x", "c"])
    assert sim == pytest.approx(1 - 1 / 3)
