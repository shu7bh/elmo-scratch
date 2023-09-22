# %%
cfg = {
    'dev_train_len': 25*10**3,
    'dev_validation_len': 5*10**3,
    'learning_rate': 0.001,
    'epochs': 100,
    'embedding_dim': 100,
    'batch_size': 32,
    'dropout': 0.2,
    'optimizer': 'Adam',
    'num_layers': 2
}

cfg['hidden_dim'] = cfg['embedding_dim']

# %%
DEV_TRAIN_LEN = cfg['dev_train_len']
DEV_VALIDATION_LEN = cfg['dev_validation_len']
LEARNING_RATE = cfg['learning_rate']
EPOCHS = cfg['epochs']
BATCH_SIZE = cfg['batch_size']
DROPOUT = cfg['dropout']
OPTIMIZER = cfg['optimizer']
NUM_LAYERS = cfg['num_layers']
HIDDEN_DIM = cfg['hidden_dim']
EMBEDDING_DIM = cfg['embedding_dim']

DIR = '/scratch/shu7bh/RES/Word/1'

# %%
import wandb
wandb.init(project="ELMo", name="WordEmb", config=cfg)

# %% [markdown]
# Create Dir

# %%
import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

# %% [markdown]
# Set Device

# %%
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# %% [markdown]
# Prepare Data

# %%
from gensim.downloader import load

glove_dict = {
    '50': 'glove-wiki-gigaword-50',
    '100': 'glove-wiki-gigaword-100',
    '200': 'glove-wiki-gigaword-200'
}

glove_dict[str(EMBEDDING_DIM)] = load(glove_dict[str(EMBEDDING_DIM)])
glove = glove_dict[str(EMBEDDING_DIM)]

# %%
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import unicodedata
import re

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)

def clean_data(text: str) -> str:
    text = normalize_unicode(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return text

lemmatizer = WordNetLemmatizer()
freq_words = dict()

def tokenize_data(text: str, create_unique_words: bool) -> list:
    tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(text)]
    tokens = [token if token in glove else '<unk>' for token in tokens]

    if '<unk>' in tokens:
        return tokens

    if create_unique_words:
        for token in tokens:
            if token not in freq_words:
                freq_words[token] = 1
            else:
                freq_words[token] += 1
    return tokens

def replace_words(tokens: list, filter_rare_words: bool) -> list:
    tokens = [token if token in freq_words else '<unk>' for token in tokens]
    if filter_rare_words:
        tokens = [token if freq_words[token] >= 4 else '<unk>' for token in tokens]
    return tokens

def read_data(path: str, create_unique_words, filter_rare_words) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    df['Description'] = df['Description'].apply(clean_data)
    df['Description'] = df['Description'].apply(tokenize_data, create_unique_words=create_unique_words)

    df = df[df['Description'].apply(lambda x: '<unk>' not in x)]
    df = df.reset_index(drop=True)

    df['Class Index'] = df['Class Index'].apply(lambda x: x-1)
    ydf = df.copy(deep=True)
    ydf['Description'] = ydf['Description'].apply(replace_words, filter_rare_words=filter_rare_words)
    return df, ydf

# %%
freq_words = dict()
xdf, ydf = read_data(
    'data/train.csv', 
    create_unique_words=True, 
    filter_rare_words=True
)

unique_words = set()
for tokens in ydf['Description']:
    unique_words.update(tokens)
print(len(unique_words))

freq_words = dict()
for token in unique_words:
    freq_words[token] = 0

# %%
xdf

# %%
NUM_CLASSES = len(set(xdf['Class Index']))
NUM_CLASSES

# %%
# Create a dictionary of all words
word_to_idx = {word: idx + 1 for idx, word in enumerate(unique_words)}

# Add special tokens
word_to_idx['<pad>'] = 0
word_to_idx['<sos>'] = len(word_to_idx)
word_to_idx['<eos>'] = len(word_to_idx)

# Create a dictionary of all words
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# print the length of the word to index mapping
print(len(word_to_idx))

# %%
idx_to_vec = glove.vectors

# %%
glove_key_to_idx = glove.key_to_index

# %%
glove_key_to_idx['<pad>'] = len(glove_key_to_idx)
glove_key_to_idx['<sos>'] = len(glove_key_to_idx)
glove_key_to_idx['<eos>'] = len(glove_key_to_idx)
glove_key_to_idx['<unk>'] = len(glove_key_to_idx)

# %%
glove_idx_to_key = {idx: key for key, idx in glove_key_to_idx.items()}

# %%
import numpy as np

glove_idx_to_vec = glove.vectors

pad_vec = np.zeros((1, EMBEDDING_DIM))
sos_vec = np.random.rand(1, EMBEDDING_DIM)
eos_vec = np.random.rand(1, EMBEDDING_DIM)
unk_vec = np.mean(glove_idx_to_vec, axis=0, keepdims=True)

glove_idx_to_vec = np.concatenate((glove_idx_to_vec, pad_vec, sos_vec, eos_vec, unk_vec), axis=0)

# %%
dev_train_raw_x = xdf[:DEV_TRAIN_LEN]
dev_train_raw_y = ydf[:DEV_TRAIN_LEN]

dev_validation_raw_x = xdf[DEV_TRAIN_LEN:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]
dev_validation_raw_y = ydf[DEV_TRAIN_LEN:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]

# %% [markdown]
# Dataset

# %%
from torch.utils.data import Dataset

class Sentences(Dataset):
    def __init__(
            self, 
            adf: pd.DataFrame, 
            pdf: pd.DataFrame, 
            word_to_idx: dict,
            glove_key_to_idx: dict
        ) -> None:

        self.Xf = []
        self.Xb = []
        self.Yf = []
        self.Yb = []
        self.L = []

        for sentence in adf['Description']:
            self.Xf += [torch.tensor(
                [glove_key_to_idx[w] for w in sentence] + 
                [glove_key_to_idx['<eos>']]
            )]
            self.Xb += [torch.tensor(
                [glove_key_to_idx['<eos>']] + 
                [glove_key_to_idx[w] for w in reversed(sentence)]
            )]

            self.L += [torch.tensor(len(sentence) + 1)]

        for sentence in pdf['Description']:
            self.Yf += [torch.tensor(
                [word_to_idx[w] for w in sentence] + 
                [word_to_idx['<eos>']] + 
                [word_to_idx['<pad>']]
            )]

            self.Yf[-1] = self.Yf[-1][1:]

            self.Yb += [torch.tensor(
                [word_to_idx[w] for w in reversed(sentence)] +
                [word_to_idx['<sos>']]
            )]

        self.Y = torch.tensor(adf['Class Index'].tolist())

    def __len__(self) -> int:
        return len(self.Xf)

    def __getitem__(self, idx: int) -> tuple:
        return self.Xf[idx], self.Xb[idx], self.Y[idx], self.L[idx], self.Yf[idx], self.Yb[idx]

# %% [markdown]
# Create Dataset

# %%
dev_train_dataset = Sentences(dev_train_raw_x, dev_train_raw_y, word_to_idx, glove_key_to_idx)
dev_validation_dataset = Sentences(dev_validation_raw_x, dev_validation_raw_y, word_to_idx, glove_key_to_idx)

# %%
def collate_fn(batch: list) -> tuple:
    xf, xb, y, l, yf, yb = zip(*batch)

    xf = torch.nn.utils.rnn.pad_sequence(xf, padding_value=glove_key_to_idx['<pad>'], batch_first=True)
    xb = torch.nn.utils.rnn.pad_sequence(xb, padding_value=glove_key_to_idx['<pad>'], batch_first=True)
    yf = torch.nn.utils.rnn.pad_sequence(yf, padding_value=word_to_idx['<pad>'], batch_first=True)
    yb = torch.nn.utils.rnn.pad_sequence(yb, padding_value=word_to_idx['<pad>'], batch_first=True)
    return xf, xb, torch.stack(y), torch.stack(l), yf, yb

# %%
from torch.utils.data import DataLoader

dev_train_loader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_validation_loader = DataLoader(dev_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# %%
type(idx_to_word)

# %% [markdown]
# ELMo

# %%
from typing import Any, Mapping
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ELMo(nn.Module):
    def __init__(
            self, 
            glove_idx_to_vec: np.ndarray,
            idx_to_word: dict,
            dropout: float, 
            num_layers: int, 
            hidden_dim: int, 
            word_embed_dim: int,
            filename: str = None
        ) -> None:

        super(ELMo, self).__init__()

        self.word_embed = nn.Embedding.from_pretrained(torch.from_numpy(glove_idx_to_vec).float(), padding_idx=glove_key_to_idx['<pad>'])

        self.lstmf = nn.LSTM(
            input_size=word_embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=True
        )

        self.lstmb = nn.LSTM(
            input_size=word_embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        del self.state_dict()['word_embed.weight']

        if filename:
            self.load_state_dict(torch.load(filename), strict=False)

    def forward(self, xf: torch.Tensor, xb: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        xf = self.word_embed(xf)
        xb = self.word_embed(xb)

        input = xf.detach().clone()

        xf = pack_padded_sequence(xf, lengths=l, batch_first=True, enforce_sorted=False)

        xb = pack_padded_sequence(xb, lengths=l, batch_first=True, enforce_sorted=False)

        xf, (hsf, csf) = self.lstmf(xf)
        xb, (hsb, csb) = self.lstmb(xb)

        xf, _ = pad_packed_sequence(xf, batch_first=True)
        xb, _ = pad_packed_sequence(xb, batch_first=True)

        xf = self.dropout(xf)
        xb = self.dropout(xb)

        return xf, xb, input, (hsf, csf), (hsb, csb)

# %% [markdown]
# Early Stopping

# %%
import numpy as np

class EarlyStopping:
    def __init__(self, patience:int = 3, delta:float = 0.001):
        self.patience = patience
        self.counter = 0
        self.best_loss:float = np.inf
        self.best_model_pth = 0
        self.delta = delta

    def __call__(self, loss, epoch: int):
        should_stop = False

        if loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.counter > self.patience:
                should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
            self.best_model_pth = epoch
        return should_stop

# %% [markdown]
# LM

# %%
from tqdm import tqdm

class LM(nn.Module):
    def __init__(self, 
            hidden_dim: int, 
            vocab_size: int, 
            filename: str = None
        ) -> None:

        super(LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.elmo = ELMo(
            glove_idx_to_vec=glove_idx_to_vec,
            idx_to_word=idx_to_word,
            dropout=DROPOUT, 
            num_layers=NUM_LAYERS, 
            hidden_dim=HIDDEN_DIM,
            word_embed_dim=EMBEDDING_DIM
        )
        self.linear_forward = nn.Linear(hidden_dim, vocab_size)
        self.linear_backward = nn.Linear(hidden_dim, vocab_size)

        if filename:
            self.load_state_dict(torch.load(filename))

    def forward(self, xf: torch.Tensor, xb: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        xf, xb, _, _, _ = self.elmo(xf, xb, l)
        yf = self.linear_forward(xf)
        yb = self.linear_backward(xb)
        return yf, yb

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int, learning_rate: float, filename: str) -> None:
        self.es = EarlyStopping()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])

        for epoch in range(epochs):
            print('----------------------------------------')
            self._train(train_loader)
            loss = self._evaluate(validation_loader)
            print(f'Epoch: {epoch + 1} | Loss: {loss:7.4f}')
            if self.es(loss, epoch):
                break
            if self.es.counter == 0:
                torch.save(self.elmo.state_dict(), os.path.join(DIR, f'{filename}_elmo.pth'))

    def _call(self, xf: torch.Tensor, xb: torch.Tensor, y: torch.Tensor, l: torch.Tensor, yf: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        xf, xb, y, yf, yb = xf.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)

        yf_hat, yb_hat = self(xf, xb, l)

        yf_hat = yf_hat.view(-1, self.vocab_size)
        yb_hat = yb_hat.view(-1, self.vocab_size)

        yf = yf.view(-1)
        yb = yb.view(-1)

        loss1 = self.criterion(yf_hat, yf)
        loss2 = self.criterion(yb_hat, yb)

        loss = (loss1 + loss2) / 2

        return loss, loss1, loss2

    def _train(self, train_loader: DataLoader) -> None:
        self.train()
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []

        pbar = tqdm(train_loader)
        for xf, xb, y, l, yf, yb in pbar:

            loss, loss1, loss2 = self._call(xf, xb, y, l, yf, yb)
            epoch_loss.append(loss.item())
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'T Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Avg Loss1: {np.mean(epoch_loss1):7.4f}, Avg Loss2: {np.mean(epoch_loss2):7.4f}')

        wandb.log({'upstream_train_loss': np.mean(epoch_loss), 'upstream_train_lossf': np.mean(epoch_loss1), 'upstream_train_lossb': np.mean(epoch_loss2)})

    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.eval()
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []
        pbar = tqdm(validation_loader)
        with torch.no_grad():
            for xf, xb, y, l, yf, yb in pbar:
                loss, loss1, loss2 = self._call(xf, xb, y, l, yf, yb)
                epoch_loss.append(loss.item())
                epoch_loss1.append(loss1.item())
                epoch_loss2.append(loss2.item())
                pbar.set_description(f'V Loss: {epoch_loss[-1]:7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Avg Loss1: {np.mean(epoch_loss1):7.4f}, Avg Loss2: {np.mean(epoch_loss2):7.4f}, Counter: {self.es.counter}, Best Loss: {self.es.best_loss:7.4f}')

        wandb.log({'upstream_validation_loss': np.mean(epoch_loss), 'upstream_validation_lossf': np.mean(epoch_loss1), 'upstream_validation_lossb': np.mean(epoch_loss2)})
        return np.mean(epoch_loss)

# %% [markdown]
# Initialize Model

# %%
lm = LM(
    hidden_dim=HIDDEN_DIM, 
    vocab_size=len(word_to_idx), 
    filename=None
).to(DEVICE)
print(lm)

# %%
from torchinfo import summary

summary(lm, device=DEVICE)

# %%
lm.fit(dev_train_loader, dev_validation_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, filename='best')

# %% [markdown]
# ## Downstream Task

# %%
downstream_train_raw_x = xdf[DEV_TRAIN_LEN+DEV_VALIDATION_LEN:]
downstream_train_raw_y = ydf[DEV_TRAIN_LEN+DEV_VALIDATION_LEN:]

downstream_validation_raw_x = xdf[:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]
downstream_validation_raw_y = ydf[:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]

# %%
print(len(downstream_train_raw_x))
print(len(downstream_validation_raw_x))

# %%
downstream_train_dataset = Sentences(downstream_train_raw_x, downstream_train_raw_y, word_to_idx, glove_key_to_idx)
downstream_validation_dataset = Sentences(downstream_validation_raw_x, downstream_validation_raw_y, word_to_idx, glove_key_to_idx)

downstream_train_loader = DataLoader(downstream_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
downstream_validation_loader = DataLoader(downstream_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# %%
from sklearn.metrics import classification_report, confusion_matrix

class NewsClassification(nn.Module):
    def __init__(self, 
            hidden_dim: int, 
            vocab_size: int, 
            num_classes: int,
            filename: str = None
        ) -> None:

        super(NewsClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.elmo = ELMo(
            glove_idx_to_vec=glove_idx_to_vec,
            idx_to_word=idx_to_word,
            dropout=DROPOUT, 
            num_layers=NUM_LAYERS, 
            hidden_dim=HIDDEN_DIM,
            word_embed_dim=EMBEDDING_DIM,
            filename=filename
        )

        for param in self.elmo.parameters():
            param.requires_grad = False

        self.delta = nn.Parameter(torch.randn(1, 3))
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, xf: torch.Tensor, xb: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        _, _, input, (hsf, csf), (hsb, csb) = self.elmo(xf, xb, l)
        hsf = hsf.permute(1, 0, 2)
        csf = csf.permute(1, 0, 2)
        hsb = hsb.permute(1, 0, 2)
        csb = csb.permute(1, 0, 2)

        hs = torch.cat([hsf, hsb], dim=2)
        cs = torch.cat([csf, csb], dim=2)

        val = (hs + cs) / 2

        input = torch.mean(input, dim=1)
        input = torch.cat([input] * val.shape[1], dim=1).unsqueeze(1)

        val = torch.cat([input, val], dim=1)
        val = (self.delta / (torch.sum(self.delta))) @ val
        val = val.squeeze()

        x = self.linear(val)
        return x

    def fit(self, 
            train_loader: DataLoader, 
            validation_loader: DataLoader, 
            epochs: int, 
            learning_rate: float
        ) -> None:

        self.es = EarlyStopping()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print('----------------------------------------')
            self._train(train_loader)
            loss = self._evaluate(validation_loader)
            print(f'Epoch: {epoch + 1} | Loss: {loss:7.4f}')
            if self.es(loss, epoch):
                break
            if self.es.counter == 0:
                torch.save(self.state_dict(), os.path.join(DIR, 'best.pth'))

    def _call(self, xf: torch.Tensor, xb: torch.Tensor, y: torch.Tensor, l: torch.Tensor, yf: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        xf, xb, y, yf, yb = xf.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)

        y_hat = self(xf, xb, l)
        y_hat = y_hat.view(-1, self.num_classes)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)

        return loss

    def _train(self, train_loader: DataLoader) -> None:
        self.train()
        epoch_loss = []
        pbar = tqdm(train_loader)
        for xf, xb, y, l, yf, yb in pbar:
            loss = self._call(xf, xb, y, l, yf, yb)
            epoch_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'T Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}')

        wandb.log({'downstream_train_loss': np.mean(epoch_loss)})

    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.eval()
        epoch_loss = []
        pbar = tqdm(validation_loader)
        with torch.no_grad():
            for xf, xb, y, l, yf, yb in pbar:
                loss = self._call(xf, xb, y, l, yf, yb)
                epoch_loss.append(loss.item())
                pbar.set_description(f'V Loss: {epoch_loss[-1]:7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Counter: {self.es.counter}, Best Loss: {self.es.best_loss:7.4f}')

        wandb.log({'downstream_validation_loss': np.mean(epoch_loss)})
        return np.mean(epoch_loss)

    def _metrics(self, test_loader: DataLoader) -> None:
        self.eval()
        self.criterion = nn.CrossEntropyLoss()
        pbar = tqdm(test_loader)
        y_pred = []
        y_true = []
        epoch_loss = []

        with torch.no_grad():
            for xf, xb, y, l, yf, yb in pbar:
                xf, xb, y, yf, yb = xf.to(DEVICE), xb.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)

                y_hat = self(xf, xb, l)
                y_hat = y_hat.view(-1, self.num_classes)
                y = y.view(-1)
                loss = self.criterion(y_hat, y)

                epoch_loss.append(loss.item())
                y_hat = torch.argmax(y_hat, dim=1)
                y_pred += y_hat.tolist()
                y_true += y.tolist()

        wandb.log({'downstrea_delta': self.delta.tolist()})

        wandb.log({'downstream_test_loss': np.mean(epoch_loss)})
        print(f'Test Loss: {np.mean(epoch_loss):7.4f}')

        cr = classification_report(y_true, y_pred, digits=4)
        wandb.log({'classification_report': cr})
        print('Classification Report:', cr)

        cm = confusion_matrix(y_true, y_pred)
        wandb.log({'confusion_matrix': cm})
        print('Confusion Matrix:', cm)


# %%
test_xdf, test_ydf = read_data('data/test.csv', create_unique_words=False, filter_rare_words=False)

# %%
downstream_test_dataset = Sentences(test_xdf, test_ydf, word_to_idx, glove_key_to_idx)
downstream_test_loader = DataLoader(downstream_test_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

# %%
delta = [None, [3, 0, 0], [1, 2, 3], [0, 0, 3], [1, 1, 1]] 
for i in range(1, len(delta)):
    delta[i] = torch.tensor(delta[i], dtype=torch.float32).reshape(1, -1).to(DEVICE)
    print(delta[i])

for d in delta:
    nc = NewsClassification(hidden_dim=HIDDEN_DIM, 
                            vocab_size=len(word_to_idx), 
                            num_classes=NUM_CLASSES,
                            filename=os.path.join(DIR, 'best_elmo.pth')
                        ).to(DEVICE)

    if d is not None:
        print(d)
        nc.delta = nn.Parameter(d, requires_grad=False)

    nc.fit(downstream_train_loader, downstream_validation_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    nc.load_state_dict(torch.load(os.path.join(DIR, 'best.pth')))
    nc._metrics(downstream_test_loader)

# %%
wandb.finish()

# %%



