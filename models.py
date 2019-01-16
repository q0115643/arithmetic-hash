import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch
import string
import numpy as np


class CharLSTM(nn.ModuleList):
    def __init__(self, alphabet_size, hidden_dim, dropout1=0.2, dropout2=0, dropout3=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.alphabet_size = alphabet_size
        self.dropout_to_LSTM = nn.Dropout(dropout1)
        self.rnn = nn.LSTM(input_size=alphabet_size, hidden_size=hidden_dim,
                           num_layers=1, dropout=dropout2, batch_first=True, bidirectional=False)
        self.dropout_to_linear_layer = nn.Dropout(dropout3)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=alphabet_size)
        
    def forward(self, inputs, lengths):
        total_length = inputs.size(1)
        embedded_input = self.dropout_to_LSTM(inputs)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        self.rnn.flatten_parameters()
        packed_sorted_output, hidden = self.rnn(packed_input)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True, total_length=total_length)
        output = sorted_output[input_unsort_indices]
        output = self.fc(output)
        output = F.log_softmax(output, dim=2)
        return output
    
    def forward2(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden) #(batch, seq_len, input_size) -> (1, 1, input_size)
        output = self.fc(output)
        output = F.softmax(output, dim=2)
        return output, hidden
    
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))


class Markov:
    """An nth-Order Markov Chain class with some lexical processing elements."""
    def __init__(self, order):
        """Initialized with a delimiting character (usually a space) and the order of the Markov chain."""
        self.states = {}
        if order > 0:
            self.order = order
        else:
            raise Exception('Markov Chain order cannot be negative or zero.')
        self.states[self.init_chain()] = self.empty_counter()
        
    def init_chain(self):
        """Helper function to generate the correct initial chain value."""
        init = []
        for i in range(self.order):
            init.append('');
        return tuple(init)
    
    def empty_counter(self):
        empty_cnt = {}
        alphabets = list(string.ascii_lowercase)
        for a in alphabets:
            empty_cnt[a] = 0
        return empty_cnt

    def step(self, a, e):
        """Helper function that pops the end of tuple 'a' and tags on str 'e'."""
        d = a[1:] + (e,)
        return d
    
    def learn(self, token_counter):
        """Adds states to the dictionary; works best with sentences."""
        for tok, cnt in token_counter:
            prev = self.init_chain()
            for c in tok:
                if prev not in self.states:
                    self.states[prev] = self.empty_counter()
                curr = self.step(prev, c)
                self.states[prev][c] += cnt
                prev = curr
    
    def get_probs(self, prefix):
        request = self.init_chain()
        if len(prefix) > 0:
            for c in prefix:
                request = self.step(request, c)
        char_counter = self.states[request]
        char_probs = []
        total_cnt = 0
        for tok, cnt in char_counter.items():
            total_cnt += cnt
        for tok, cnt in char_counter.items():
            prob = float(cnt)/float(total_cnt)
            char_probs.append(tuple([tok, prob]))
        return char_probs

class RNN_Map():
    def __init__(self, model_path, num_node, using_GPU, hidden_dim=64):
        alphabets = list(string.ascii_lowercase)
        self.alphabet_size = len(alphabets) + 1
        self.int2char = dict(enumerate(alphabets, start=1))
        self.int2char[0] = '<PAD>'
        self.char2int = {char: index for index, char in self.int2char.items()}
        self.RNN_model = CharLSTM(alphabet_size=self.alphabet_size, hidden_dim=hidden_dim)
        self.using_GPU = using_GPU
        if self.using_GPU:
            self.RNN_model = self.RNN_model.cuda()
        state = torch.load(model_path)
        self.RNN_model.load_state_dict(state['model'])
        self.RNN_model.eval()
        self.num_node = num_node
    
    def get_pos(self, token):
        assert len(token) > 0
        hidden = self.RNN_model.init_hidden()
        pos = 0
        sec_len = 1
        prev = torch.stack([torch.Tensor(np.zeros((1, self.alphabet_size)))])
        for c in token:
            with torch.no_grad():
                prev = Variable(prev)
                if self.using_GPU:
                    prev = prev.cuda()
                    hidden = (hidden[0].cuda(), hidden[1].cuda())
                output, hidden = self.RNN_model.forward2(prev, hidden)
                prev = [torch.Tensor(np.zeros((1, self.alphabet_size)))]
                prev[0][0][self.char2int[c]] = 1
                prev = torch.stack(prev)
            for idx, prob in enumerate(list(output.cpu().numpy()[0][0])):
                if self.int2char[idx] == c:
                    sec_len = prob * sec_len
                    break
                else:
                    pos += prob * sec_len
        return pos
    
    def get_node(self, token):
        sec_len = 1. / self.num_node
        pos = self.get_pos(token)
        return 1 + int(pos // sec_len)
