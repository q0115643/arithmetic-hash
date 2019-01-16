from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from util import list_to_string, only_alphabets
import csv
import sys
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', stream=sys.stdout)
brown_fp = './data/corpora/brown.txt'
train_tokens_fp = './data/train_tokens.txt'
test_tokens_fp = './data/test_tokens.txt'

tokens = []
logging.info('Tokenizing Brown Corpus...')
with open(brown_fp, 'r') as brown:
    brown = list(brown)
    for line in tqdm(brown, total=len(brown)):
        words = word_tokenize(only_alphabets(line))
        tokens += words
logging.info("Writing Token-Count info on new txt file...")
train_tokens, test_tokens = train_test_split(tokens, test_size=0.1)
with open(train_tokens_fp, 'w') as writeFile:
    train_tokens = list(set(train_tokens))
    train_tokens = sorted(train_tokens, key = lambda s : s.lower())
    for token in train_tokens:
        writeFile.write("%s\n" % token)
with open(test_tokens_fp, 'w') as writeFile:
    test_tokens = list(set(test_tokens))
    test_tokens = sorted(test_tokens, key = lambda s : s.lower())
    for token in test_tokens:
        writeFile.write("%s\n" % token)
