import string
import unittest

import numpy as np

from cryptogram import (
    _get_unique_token_sequence,
    get_shared_token_sequence_mask,
    get_shared_word_length_mask,
    vectorize_text,
    word_vector_overlap
)


class Test(unittest.TestCase):
    def test_vectorize_text_no_max_len(self):
        text = [
            'one',
            'ring',
            'to',
        ]
        alphabet = 'onerigtabcd'

        actual = vectorize_text(text, alphabet)
        expected = np.array([
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ])

        np.testing.assert_array_equal(actual, expected)

    @unittest.skip('')
    def test_vectorize_text_no_max_len2(self):
        text = [
            'one',
            'ring',
            'to',
            'rule',
            'them',
            'all',
            'rings',
            'top',
            'tip',
            'ale'
        ]
        alphabet = 'onerigtulhmasp'

        actual = vectorize_text(text, alphabet)
        expected = np.array([
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ])

        np.testing.assert_array_equal(actual, expected)

    def test_vectorize_text_with_max_len(self):
        text = [
            'one',
            'ring',
            'to',
        ]
        alphabet = 'onerigtabcd'

        actual = vectorize_text(text, alphabet, 2)
        expected = np.array([
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        ])

        np.testing.assert_array_equal(actual, expected)

    def test_word_vector_overlap(self):
        X = vectorize_text([
            'one',
            'ring',
            'to',
        ], 'onerigtulhmasp')
        X = np.pad(X, [(0, 0), (0, 1), (0, 0)])
        V = vectorize_text([
            'one',
            'ring',
            'to',
            'rule',
            'them',
            'all',
            'rings',
            'top',
            'tip',
            'ale'
        ], 'onerigtulhmasp')

        actual = word_vector_overlap(X, V)
        expected = np.array([
            [3, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 4, 0, 1, 0, 0, 4, 0, 1, 0],
            [0, 0, 2, 0, 1, 0, 0, 2, 1, 0],
        ])

        np.testing.assert_array_equal(actual, expected)

    def test_get_mask_same_alphabet(self):
        x = vectorize_text([
            'molly',
            'folly',
            'to',
            'sully',
            'artsy',
            'on'
        ], string.ascii_lowercase)
        y = vectorize_text([
            'fully',
            'party',
            'no',
            'artsy'
        ], string.ascii_lowercase)

        actual = get_shared_token_sequence_mask(x, y)
        expected = np.array([
            [True,  False, False, False],
            [True,  False, False, False],
            [False, False,  True, False],
            [True,  False, False, False],
            [False,  True, False,  True],
            [False, False,  True, False],
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_get_mask_different_alphabets(self):
        # because the comparison is done after we aggregate over the alphabet
        # dimension up to the word position, the underlying dimensionality shouldn't
        # matter here
        x = vectorize_text([
            'molly',
            'folly',
            'to',
            'sully',
            'artsy',
            'on',
        ], 'molyftsuarn')
        y = vectorize_text([
            'fully',
            'party',
            'no',
            'artsy',
        ], string.ascii_lowercase)

        actual = get_shared_token_sequence_mask(x, y)
        expected = np.array([
            [True,  False, False, False],
            [True,  False, False, False],
            [False, False,  True, False],
            [True,  False, False, False],
            [False,  True, False,  True],
            [False, False,  True, False],
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_get_shared_word_length_mask(self):
        # because the comparison is done after we aggregate over the alphabet
        # dimension up to the word position, the underlying dimensionality shouldn't
        # matter here
        x = [
            'molly',
            'folly',
            'to',
            'sully',
            'artsy',
            'on',
        ]
        y = [
            'fully',
            'party',
            'no',
            'artsy',
        ]

        actual = get_shared_word_length_mask(x, y)
        expected = np.array([
            [True,   True, False,  True],
            [True,   True, False,  True],
            [False, False,  True, False],
            [True,   True, False,  True],
            [True,   True, False,  True],
            [False, False,  True, False],
        ])
        np.testing.assert_array_equal(actual, expected)

    def test__get_unique_token_sequence(self):
        X = vectorize_text([
            'one',
            'ring',
            'to',
            'too',
            'tool',
            'rings',
        ], 'onerigtulhmaspls')

        actual = _get_unique_token_sequence(X)
        expected = np.array([
            [1, 2, 3, 3, 3],
            [1, 2, 3, 4, 4],
            [1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2],
            [1, 2, 2, 3, 3],
            [1, 2, 3, 4, 5],
        ])

        np.testing.assert_array_equal(actual, expected)
