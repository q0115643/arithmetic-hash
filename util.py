import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import math


def fp_to_list(fp):  
    with open(fp) as f:
        tokens = f.readlines()
        tokens = [x.strip() for x in tokens]
    return tokens

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

def evaluate(model, dataloader, criterion, alphabet_size, using_GPU):
    model.eval()
    total_eval_loss = 0
    corrects = 0
    total_len = 0
    for (inputs, targets, lengths) in dataloader:
        with torch.no_grad():
            inputs = Variable(inputs) # shape(batch_size, longest_length, alphabet_num) (ex. 128, 13, 28)
            targets = Variable(targets)
            lengths = Variable(lengths)
            if using_GPU:
                inputs = inputs.cuda() # [128, maxlen, 26]
                targets = targets.cuda()
                lengths = lengths.cuda()
            predicted = model(inputs, lengths)
            total_eval_loss += criterion(predicted.view(-1, alphabet_size), targets.view(-1)).item()
            total_len += predicted.view(-1, alphabet_size).shape[0]
            corrects += (torch.max(predicted.view(-1, alphabet_size), 1)[1].data == targets.view(-1).data).sum()
    avg_eval_loss = total_eval_loss / dataloader.__len__()
    accuracy = float(corrects) / float(total_len)
    model.train()
    return avg_eval_loss, accuracy

class CharDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __getitem__(self, idx):
        example_token = self.tokens[idx]
        # Truncate the sequence if necessary
        example_length = example_token.shape[0]
        return example_token, example_length

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def collate_fn(batch):
        batch_inputs = []
        batch_padded_example_text = []
        batch_lengths = []
        batch_targets = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for token, __ in batch:
            if len(token) > max_length:
                max_length = len(token)

        # Iterate over each example in the batch
        for token, length in batch:
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, token.shape[1])
            batch_targets.append([np.where(x == 1)[0][0] for x in np.array(token[1:])] + [0] * amount_to_pad)
            token = torch.Tensor(token)
            padded_example_text = torch.cat((token, pad_tensor), dim=0)
            # Add the padded example to our batch
            batch_lengths.append(length-1)
            batch_inputs.append(padded_example_text.narrow(0, 0, max_length-1))

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_inputs),
                torch.LongTensor(batch_targets),
                torch.LongTensor(batch_lengths))

def get_stddev(num_node, num_token, cnt_per_node):
    avg_node = num_token/float(num_node)
    variance = 0
    for cnt in cnt_per_node:
        diff = cnt - avg_node
        variance += diff*diff
    std_dev = math.sqrt(variance/num_token)
    return std_dev