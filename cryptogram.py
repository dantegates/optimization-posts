import numpy as np
import torch

def vectorize_text(text, alphabet, max_len=None):
    max_vocab_len = max(len(w) for w in text)
    max_len = min(max_vocab_len, max_len or max_vocab_len)

    index_word = np.vectorize(alphabet.index)
    X = np.zeros((len(text), max_len, len(alphabet)))
    for xi, word in zip(X, text):
        word = word[:max_len]
        xi[np.arange(len(word)), index_word(list(word))] = 1

    return X

def word_vector_overlap(X, V):
    X = torch.from_numpy(X)
    V = torch.from_numpy(V)
    return torch.tensordot(X, V, dims=[[1, 2], [1, 2]]).numpy()

def get_shared_word_length_mask(text1, text2):
    word_length = np.vectorize(len)
    return word_length(text1)[:, None] == word_length(text2)[None, :]

def get_shared_word_separater_mask(text1, text2):
    separater_pos = np.vectorize(len)
    return word_length(text1)[:, None] == word_length(text2)[None, :]

def get_shared_token_sequence_mask(x1, x2):
    s1, s2 = _get_unique_token_sequence(x1), _get_unique_token_sequence(x2)
    return (s1[:, None, :] == s2).all(axis=-1)

def _get_unique_token_sequence(x):
    return np.maximum.accumulate(x, axis=1).sum(axis=2)
