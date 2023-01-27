import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np

import sys
sys.path.append('../')
from attention.multihead_self import ScaledDotProductAttention

class Label_Attention(nn.Module):

    def __init__(self, batch_size, label_size, embedding_size):
        super(Label_Attention, self).__init__()
        self.label_size = label_size
        self.batch_size = batch_size

        self.label_representation = nn.Parameter((torch.Tensor(label_size, embedding_size)))

        nn.init.xavier_normal_(self.label_representation)

    def forward(self, sentence_embeddings):
        sentence_embeddings = torch.tensor(sentence_embeddings)
        context, attn = ScaledDotProductAttention(1)(sentence_embeddings, self.label_representation.expand(self.batch_size, *self.label_representation.shape), sentence_embeddings)

        return context


class BiLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        hidden_layer, (h_1, c_1) = self.lstm(input)

        return hidden_layer

class GCN_v1(nn.Module):
    @staticmethod
    def normalizeAdjacency(W):
        """
        NormalizeAdjacency: Computes the degree-normalized adjacency matrix

        Input:

            W (np.array): adjacency matrix

        Output:

            A (np.array): degree-normalized adjacency matrix
        """
        # Check that the matrix is square
        assert W.shape[0] == W.shape[1]
        # Compute the degree vector
        d = torch.sum(W, dim=1)
        # Invert the square root of the degree
        d = 1/torch.sqrt(d)
        # And build the square root inverse degree matrix
        D = torch.diag(d)
        # Return the Normalized Adjacency
        return torch.matmul(torch.matmul(D, W), D)

    def __init__(self, batch_size, input_size, hidden_size):
        super(GCN_v1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.weight_1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.leaky_relu = nn.LeakyReLU()
        
        torch.nn.init.xavier_normal_(self.weight_1)
        torch.nn.init.xavier_normal_(self.weight_2)
        
    def compute_gcn_layer(self, adj_mat, prev, weight):
        intermediate = torch.bmm(adj_mat, prev)
        propagate = torch.bmm(intermediate, weight)
        return self.leaky_relu(propagate)

    def forward(self, adj_matrix, input):
        self.normalize_adj_mat = GCN_v1.normalizeAdjacency(adj_matrix).float().expand(self.batch_size, *adj_matrix.shape)

        h_1 = self.compute_gcn_layer(self.normalize_adj_mat, input, self.weight_1.expand(self.batch_size, self.input_size, self.hidden_size))
        h_2 = self.compute_gcn_layer(self.normalize_adj_mat, h_1, self.weight_2.expand(self.batch_size, self.hidden_size, self.hidden_size))
        return h_2

class Dynamic(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Dynamic, self).__init__()
        self.weight_a = nn.Parameter(torch.Tensor(1, hidden_size))
        self.weight_b = nn.Parameter(torch.Tensor(1, hidden_size))
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.weight_a)
        nn.init.xavier_normal_(self.weight_b)

    def forward(self, H):
        conv_1 = torch.sum(self.weight_a * H, dim=-1).unsqueeze(0)
        conv_2 = torch.sum(self.weight_b * H, dim=-1).unsqueeze(0)
        propagate = torch.bmm(conv_1.transpose(-1, -2), conv_2)
        return self.sigmoid(propagate)

class LDGB(torch.nn.Module):
    def __init__(self, batch_size, input_size, embedding_size, hidden_size, label_size, embedding):
        super(LDGB, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float(), freeze=True)
        self.lstm = BiLSTM(1, embedding_size, embedding_size, 0.2)
        self.label_attention = Label_Attention(batch_size, label_size, embedding_size)
        self.gcn_1 = GCN_v1(batch_size, embedding_size, hidden_size)
        self.dynamic = Dynamic(hidden_size)
        self.gcn_2 = GCN_v1(batch_size, hidden_size, hidden_size)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, input):
        input_embeddings = self.embedding(input) # (batch_size, seq_len, embedding_size)

        lstm_result = self.lstm(input_embeddings, (None, None)).reshape(self.batch_size, self.input_size*2, self.embedding_size) # (batch_size, 2*seq_len, embedding_size)

        h_0 = self.label_attention.forward(lstm_result) # (batch_size, label_size, embedding_size)

        h_2 = self.gcn_1.forward(A, h_0) # (batch_size, label_size, gcn_hidden)
        A_new = self.dynamic.forward(h_2) # (batch_size, label_size, label_size)
        A_new = torch.mean(A_new, dim=0) # (label_size, label_size)
        h_4 = self.gcn_2.forward(A_new, h_2) # (batch_size, label_size, gcn_hidden)
        h_o = torch.concat([h_2, h_4], dim=-1) # (batch_size, label_size, gcn_hidden*2)

        return self.sigmoid(self.fc(h_o)) # (batch_size, label_size, 1)