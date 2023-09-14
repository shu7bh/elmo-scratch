import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ELMO(nn.Module):
    def __init__(self, Emb, hidden_dim, dropout, num_layers):
        super().__init__()

        self.Emb = Emb
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(Emb.vectors), padding_idx=Emb.key_to_index['<pad>'])
        self.lstm = nn.LSTM(Emb.vectors.shape[1], hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, Emb.vectors.shape[0])

    def forward(self, X, X_lengths):
        X = self.embedding(X)
        X = pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        X, _ = self.lstm(X, None)
        X, _ = pad_packed_sequence(X, batch_first=True)
        X = self.fc(X)
        return X