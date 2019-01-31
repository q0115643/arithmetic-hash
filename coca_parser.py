from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from util import list_to_string, only_alphabets
import csv
import sys
import logging
import re
import argparse
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gram_num', type=int, default=2)
    parser.add_argument('--load_model', type=bool, default=False)
    args = parser.parse_args()
    return args

args = get_args()
gram_num = args.gram_num

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', stream=sys.stdout)
coca_fp = './data/corpora/coca_' + str(gram_num) + 'gram.txt'
train_tokens_fp = './data/coca/' + str(gram_num) + 'gram/train_tokens.txt'
val_tokens_fp = './data/coca/' + str(gram_num) + 'gram/val_tokens.txt'
test_tokens_fp = './data/coca/' + str(gram_num) + 'gram/test_tokens.txt'

tokens = []
logging.info('Tokenizing Brown Corpus...')
with open(coca_fp, 'r') as coca:
    coca = list(coca)
    for line in tqdm(coca, total=len(coca)):
        items = [item.strip() for item in line.split('\t')]
        freq = int(items[0])
        if freq > 1:
            token = ''
            for i in range(gram_num):
                if i != 0:
                    token += '_'
                token += re.sub(r"\s+", "", only_alphabets(items[i+1]))
            for _ in range(freq-1):
                tokens.append(token)
logging.info("Number of Tokens: {}".format(len(tokens)))
logging.info("Number of Different Tokens: {}".format(len(list(set(tokens)))))
train_tokens, test_tokens = train_test_split(tokens, test_size=0.1)
train_tokens, val_tokens = train_test_split(train_tokens, test_size=0.1)
train_tokens = list(set(train_tokens))
val_tokens = list(set(val_tokens))
test_tokens = list(set(test_tokens))
logging.info("Writing Training Tokens info on new txt file...")
with open(train_tokens_fp, 'w') as writeFile:
    train_tokens = sorted(train_tokens, key = lambda s : s.lower())
    for token in tqdm(train_tokens):
        writeFile.write("%s\n" % token)
logging.info("Writing Validation Tokens info on new txt file...")
with open(val_tokens_fp, 'w') as writeFile:
    val_tokens = sorted(val_tokens, key = lambda s : s.lower())
    for token in tqdm(val_tokens):
        writeFile.write("%s\n" % token)
logging.info("Writing Test Tokens info on new txt file...")
with open(test_tokens_fp, 'w') as writeFile:
    test_tokens = sorted(test_tokens, key = lambda s : s.lower())
    for token in tqdm(test_tokens):
        writeFile.write("%s\n" % token)
