import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch

class CharLSTM(nn.ModuleList):
    def __init__(self, alphabet_size, hidden_dim, batch_size, dropout1=0.2, dropout2=0, dropout3=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
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