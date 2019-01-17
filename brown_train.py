import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import string
import logging
import sys
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models import CharLSTM
import torch.nn as nn
from torch.autograd import Variable
from util import to_categorical, CharDataset, evaluate

batch_size = 2048
hidden_dim = 64
dropout1 = 0.2
dropout2 = 0
dropout3 = 0.2
lr = 0.2
num_epoch = 10000
epoch_print_cycle = 1
lr_decay_rate = 0.9
load_model = True

train_tokens_fp = './data/brown/train_tokens.txt'
val_tokens_fp = './data/brown/val_tokens.txt'
test_tokens_fp = './data/brown/test_tokens.txt'
model_path = './data/checkpoint/brown/rnn.pkl'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', stream=sys.stdout)
logging.info("PyTorch version: {}".format(torch.__version__))
logging.info("GPU Detected: {}".format(torch.cuda.is_available()))
using_GPU = torch.cuda.is_available()


logging.info("*"*30)
logging.info("Data Loader")
logging.info("*"*30)
alphabets = list(string.ascii_lowercase)
alphabet_size = len(alphabets) + 2
int2char = dict(enumerate(alphabets, start=2))
int2char[0] = '<PAD>'
int2char[1] = '<END>'
char2int = {char: index for index, char in int2char.items()}
train_tokens = []
with open(train_tokens_fp) as f:
    train_tokens = f.readlines()
    train_tokens = [x.strip() for x in train_tokens] 
for t in train_tokens:
    if len(t) < 1:
        print(t)
train_tokens = [np.array([char2int[char] for char in token] + [char2int["<END>"]]) for token in train_tokens]
val_tokens = []
with open(val_tokens_fp) as f:
    val_tokens = f.readlines()
    val_tokens = [x.strip() for x in val_tokens] 
val_tokens = [np.array([char2int[char] for char in token] + [char2int["<END>"]]) for token in val_tokens]
test_tokens = []
with open(test_tokens_fp) as f:
    test_tokens = f.readlines()
    test_tokens = [x.strip() for x in test_tokens] 
test_tokens = [np.array([char2int[char] for char in token] + [char2int["<END>"]]) for token in test_tokens]
encoded_train_tokens = []
logging.info('One-hot Encoding Train Tokens...')
for token in tqdm(train_tokens):
    encoded_train_tokens.append(to_categorical(token, alphabet_size))
encoded_val_tokens = []
logging.info('One-hot Encoding Validation Tokens...')
for token in tqdm(val_tokens):
    encoded_val_tokens.append(to_categorical(token, alphabet_size))
encoded_test_tokens = []
logging.info('One-hot Encoding Test Tokens...')
for token in tqdm(test_tokens):
    encoded_test_tokens.append(to_categorical(token, alphabet_size))
training_dataset = CharDataset(encoded_train_tokens)
val_dataset = CharDataset(encoded_val_tokens)
test_dataset = CharDataset(encoded_test_tokens)
train_dataloader = DataLoader(dataset=training_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=CharDataset.collate_fn)
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=CharDataset.collate_fn)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=CharDataset.collate_fn)

logging.info("*"*30)
logging.info("Train Model")
logging.info("*"*30)
RNN_model = CharLSTM(alphabet_size=alphabet_size,
                     hidden_dim=hidden_dim,
                     dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)
if using_GPU:
    RNN_model = RNN_model.cuda()
if load_model:
    state = torch.load(model_path)
    eval_loss_record = state['eval_loss_record']
    train_loss_record = state['train_loss_record']
    min_val_loss = state['min_val_loss']
    last_val_loss = state['last_val_loss']
    last_train_loss = state['last_train_loss']
    RNN_model.load_state_dict(state['model'])
else:
    train_loss_record = []
    eval_loss_record = []
    min_val_loss = 9999999
    last_val_loss = 9999999
    last_train_loss = 9999999
loss_criterion = nn.NLLLoss(reduction='mean')
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=lr)
if load_model:
    optimizer.load_state_dict(state['optimizer'])
#optimizer = torch.optim.SGD(RNN_model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-5)
num_iter = 0
RNN_model.train()
logging.info('Start Training with {} Epoch'.format(num_epoch))
try:
    for epoch in range(1, num_epoch+1):
        for (inputs, targets, lengths) in train_dataloader:
            inputs = Variable(inputs) # shape(batch_size, longest_length, alphabet_num) (ex. 128, 13, 28)
            lengths = Variable(lengths)
            targets = Variable(targets)
            if using_GPU:
                inputs = inputs.cuda() # [128, maxlen, 26]
                lengths = lengths.cuda()
                targets = targets.cuda()
            output = RNN_model(inputs, lengths) # [batch_size, maxlen, hidden_dim]
            batch_loss = loss_criterion(output.view(-1, alphabet_size), targets.view(-1))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss_record.append(batch_loss.item())
            num_iter += 1
        if epoch % epoch_print_cycle == 0:
            val_loss, accuracy = evaluate(RNN_model, val_dataloader, loss_criterion, alphabet_size, using_GPU)
            eval_loss_record.append(val_loss)
            logging.info("Epoch {}. Batch Loss {:.4f}.".format(epoch, batch_loss))
            logging.info("Validation Loss {:.4f}. Validataion Accuracy {:.2f}.".format(val_loss, accuracy))
            if min_val_loss > val_loss:
                logging.info("New Best Validation Loss {:.4f} -> {:.4f}, Saving Model Checkpoint".format(min_val_loss, val_loss))
                min_val_loss = val_loss
                torch.save({'model': RNN_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'train_loss_record': train_loss_record,
                            'eval_loss_record': eval_loss_record,
                            'epoch_print_cycle': epoch_print_cycle,
                            'min_val_loss': min_val_loss,
                            'last_train_loss': last_train_loss,
                            'last_val_loss': last_val_loss
                           }, model_path)
            elif last_val_loss < val_loss and last_train_loss < batch_loss:
                new_lr = lr * lr_decay_rate
                logging.info("Learning Rate Decay {} -> {}".format(lr, new_lr))
                lr = new_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            last_val_loss = val_loss
            last_train_loss = batch_loss
except KeyboardInterrupt:
    logging.info("Stop By KeyboardInterrupt...")
