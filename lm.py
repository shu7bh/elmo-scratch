from torch import nn
from elmo import ELMO

class LM(nn.Module):
    def __init__(self, Emb, hidden_dim, dropout, num_layers):
        super().__init__()

        self.elmo = ELMO(Emb, hidden_dim, dropout, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, Emb.vectors.shape[0])

    def forward(self, X, X_lengths):
        X, _ = self.elmo(X, X_lengths)
        X = self.fc(X)
        return X