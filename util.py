import re
import numpy as np


def list_to_string(list_tokens):
    res = ''
    first = True
    for tok in list_tokens:
        if first:
            first = False
        else:
            res += ' '
        res += tok
    return res

def only_alphabets(text):
    return re.sub(r'[^a-z]+', ' ', text.lower())

def to_categorical(y, n_classes):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], n_classes))
    y_cat[range(y.shape[0]), y] = 1.
    start = np.zeros((1, n_classes))
    return np.append(start, y_cat, axis=0)
