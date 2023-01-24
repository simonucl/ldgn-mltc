import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import sys
sys.path.append('../')
from attention.multihead_self import ScaledDotProductAttention

class Label_Attention(nn.Module):

    def __init__(self, label_size, embedding_size):
        super(Label_Attention, self).__init__()
        self.label_size = label_size

        self.label_representation = torch.empty(label_size, embedding_size)

        nn.init.xavier_uniform_(self.label_representation)
    def forward(self, sentence_embeddings):
        sentence_embeddings = torch.tensor(sentence_embeddings)
        context, attn = ScaledDotProductAttention(1)(sentence_embeddings, self.label_representation, sentence_embeddings)

        return context


class BiLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        hidden_layer, (h_1, c_1) = self.lstm(input, h_0, c_0)

        return hidden_layer