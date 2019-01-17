from nltk.tokenize import word_tokenize
from tqdm import tqdm
from util import list_to_string, only_alphabets
import csv
import sys
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', stream=sys.stdout)
brown_fp = './data/corpora/brown.txt'
train_tokens_fp = './data/brown/train_tokens.txt'
val_tokens_fp = './data/brown/val_tokens.txt'
test_tokens_fp = './data/brown/test_tokens.txt'

tokens = []
logging.info('Tokenizing Brown Corpus...')
with open(brown_fp, 'r') as brown:
    brown = list(brown)
    for line in tqdm(brown, total=len(brown)):
        words = word_tokenize(only_alphabets(line))
        tokens += words
logging.info("Number of Tokens: {}".format(len(tokens)))
logging.info("Number of Different Tokens: {}".format(len(list(set(tokens)))))
train_tokens, test_tokens = train_test_split(tokens, test_size=0.1)
train_tokens, val_tokens = train_test_split(train_tokens, test_size=0.05)
logging.info("Writing Train Tokens info on new txt file...")
with open(train_tokens_fp, 'w') as writeFile:
    train_tokens = sorted(list(set(train_tokens)), key = lambda s : s.lower())
    for token in tqdm(train_tokens):
        writeFile.write("%s\n" % token)
logging.info("Writing Validation Tokens info on new txt file...")
with open(val_tokens_fp, 'w') as writeFile:
    val_tokens = sorted(list(set(val_tokens)), key = lambda s : s.lower())
    for token in tqdm(val_tokens):
        writeFile.write("%s\n" % token)
logging.info("Writing Test Tokens info on new txt file...")
with open(test_tokens_fp, 'w') as writeFile:
    test_tokens = sorted(list(set(test_tokens)), key = lambda s : s.lower())
    for token in tqdm(test_tokens):
        writeFile.write("%s\n" % token)
