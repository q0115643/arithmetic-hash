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
train_tokens_csv_fp = './data/train_tokens.csv'
test_tokens_csv_fp = './data/test_tokens.csv'

tokens = []
logging.info('Tokenizing Brown Corpus...')
with open(brown_fp, 'r') as brown:
    brown = list(brown)
    for line in tqdm(brown, total=len(brown)):
        words = word_tokenize(only_alphabets(line))
        tokens += words
logging.info("Writing Token-Count info on new csv file...")
train_tokens, test_tokens = train_test_split(tokens, test_size=0.1)
with open(train_tokens_csv_fp, 'w') as csvfile:
    fieldnames = [
        'token',
        'count'
    ]
    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames,
        quoting=csv.QUOTE_ALL)
    writer.writeheader()
    output = []
    token_counter = sorted(Counter(train_tokens).items(), key=lambda pair: pair[1], reverse=True)
    for tok, cnt in tqdm(token_counter):
        output.append({'token': tok,
                       'count': cnt})
    writer.writerows(output)
with open(test_tokens_csv_fp, 'w') as csvfile:
    fieldnames = [
        'token',
        'count'
    ]
    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames,
        quoting=csv.QUOTE_ALL)
    writer.writeheader()
    output = []
    token_counter = sorted(Counter(test_tokens).items(), key=lambda pair: pair[1], reverse=True)
    for tok, cnt in tqdm(token_counter):
        output.append({'token': tok,
                       'count': cnt})
    writer.writerows(output)
