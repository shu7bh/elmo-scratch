# %%
cfg = {
    'dev_train_len': 25*10**3,
    'dev_validation_len': 5*10**3,
    'learning_rate': 0.001,
    'epochs': 100,
    'char_embedding_dim': 32,
    'batch_size': 32,
    'dropout': 0.1,
    'optimizer': 'Adam',
    'num_layers': 2,
    'word_emb_dim': 200,
    'max_word_len': 20,
    'hidden_dim': 100,
    'char_out_channels': 64,
}

# %% [markdown]
# Hyperparameters

# %%
DEV_TRAIN_LEN = cfg['dev_train_len']
DEV_VALIDATION_LEN = cfg['dev_validation_len']
LEARNING_RATE = cfg['learning_rate']
EPOCHS = cfg['epochs']
CHAR_EMBEDDING_DIM = cfg['char_embedding_dim']
BATCH_SIZE = cfg['batch_size']
DROPOUT = cfg['dropout']
OPTIMIZER = cfg['optimizer']
NUM_LAYERS = cfg['num_layers']
HIDDEN_DIM = cfg['hidden_dim']
WORD_EMB_DIM = cfg['word_emb_dim']
MAX_WORD_LEN = cfg['max_word_len']
CHAR_OUT_CHANNELS = cfg['char_out_channels']

DIR = '/scratch/shu7bh/RES/4'

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
import os

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# %% [markdown]
# Prepare Data

# %%
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import unicodedata
import re

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)

unique_chars = set()

def clean_data(text: str) -> str:
    text = normalize_unicode(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    for char in text:
        unique_chars.add(char)
    return text

lemmatizer = WordNetLemmatizer()
freq_words = dict()
def tokenize_data(text: str, create_unique_words: bool) -> list:
    global freq_words
    tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(text)]

    if create_unique_words:
        for token in tokens:
            if token not in freq_words:
                freq_words[token] = 1
            else:
                freq_words[token] += 1
    return tokens

def replace_words(tokens: list, filter_rare_words: bool) -> list:
    new_tokens = []
    for i in range(len(tokens)):
        if tokens[i] not in freq_words:
            new_tokens.append('<unk>')
        else:
            if filter_rare_words:
                if freq_words[tokens[i]] < 4:
                    new_tokens.append('<unk>')
                else:
                    new_tokens.append(tokens[i])
    return new_tokens

def read_data(path: str, create_unique_words, filter_rare_words) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    df['Description'] = df['Description'].apply(clean_data)
    df['Description'] = df['Description'].apply(tokenize_data, create_unique_words=create_unique_words)
    df['Class Index'] = df['Class Index'].apply(lambda x: x-1)
    pred_df = df.copy(deep=True)
    pred_df['Description'] = pred_df['Description'].apply(replace_words, filter_rare_words=filter_rare_words)
    return df, pred_df

# %%
freq_words = dict()
actual_df, pred_df = read_data(
    'data/train.csv', 
    create_unique_words=True, 
    filter_rare_words=True
)

unique_words = set()
for tokens in pred_df['Description']:
    unique_words.update(tokens)
print(len(unique_words))

# %%
NUM_CLASSES = len(set(actual_df['Class Index']))
NUM_CLASSES

# %% [markdown]
# Char -> Idx and Word -> Idx

# %%
# Create a dictionary of all characters
char_to_idx = {char: idx + 1 for idx, char in enumerate(unique_chars)}

# Add special tokens
char_to_idx['<pad>'] = 0
char_to_idx['<sos>'] = len(char_to_idx)
char_to_idx['<eos>'] = len(char_to_idx)

# Create a dictionary of all characters
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# print the character to index mapping
print(char_to_idx)
print(idx_to_char)

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
dev_train_raw_a = actual_df[:DEV_TRAIN_LEN]
dev_train_raw_p = pred_df[:DEV_TRAIN_LEN]

dev_validation_raw_a = actual_df[DEV_TRAIN_LEN:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]
dev_validation_raw_p = pred_df[DEV_TRAIN_LEN:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]

# %% [markdown]
# Dataset

# %%
from torch.utils.data import Dataset

class Sentences(Dataset):
    def __init__(
            self, 
            adf: pd.DataFrame, 
            pdf: pd.DataFrame, 
            char_to_idx: dict, 
            word_to_idx: dict
        ) -> None:

        self.X = []
        self.Y_ = []

        for sentence in adf['Description']:
            sent = []
            for w in sentence:
                sent += [[char_to_idx[w[i]] for i in range(min(MAX_WORD_LEN, len(w)))] + [char_to_idx['<pad>']] * (MAX_WORD_LEN - len(w))]

            sent += [[char_to_idx['<eos>']] * MAX_WORD_LEN]
            sent = torch.cat([torch.tensor(word) for word in sent])
            self.X += [sent]

        for sentence in pdf['Description']:
            self.Y_ += [torch.tensor([word_to_idx['<sos>']] + [word_to_idx[w] for w in sentence] + [word_to_idx['<eos>']])]

        self.Y = torch.tensor(adf['Class Index'].tolist())

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx], torch.tensor(len(self.X[idx])), self.Y_[idx]

# %% [markdown]
# Create Dataset

# %%
dev_train_dataset = Sentences(dev_train_raw_a, dev_train_raw_p, char_to_idx, word_to_idx)
dev_validation_dataset = Sentences(dev_validation_raw_a, dev_validation_raw_p, char_to_idx, word_to_idx)

# %% [markdown]
# Collate

# %%
def collate_fn(batch: list) -> tuple:
    x, y, l, y_ = zip(*batch)

    x = torch.nn.utils.rnn.pad_sequence(x, padding_value=char_to_idx['<pad>'], batch_first=True)
    y_ = torch.nn.utils.rnn.pad_sequence(y_, padding_value=word_to_idx['<pad>'], batch_first=True)
    y_ = torch.cat([y_, torch.zeros(y_.shape[0], 1, dtype=torch.long)], dim=1)
    l = [i/20 for i in l]
    return x, torch.stack(y), torch.stack(l), y_[..., 2:], y_[..., :-2]

# %% [markdown]
# Create DataLoader

# %%
from torch.utils.data import DataLoader

dev_train_loader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_validation_loader = DataLoader(dev_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# %% [markdown]
# CharCNN

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(
            self, 
            char_vocab: int,
            char_embed_dim: int,
            char_out_channels: list,
            char_kernel_sizes: list,
            dropout: float,
            word_embed_dim: int
        ) -> None:

        super(CharCNN, self).__init__()

        self.char_embed = nn.Embedding(char_vocab, char_embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.conv_max_pools = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_embed_dim, char_out_channels[i], char_kernel_sizes[i]),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            for i in range(len(char_out_channels))
        ])

        self.fc = nn.Linear(sum(char_out_channels), word_embed_dim) # the fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.char_embed(x)
        x = x.transpose(1, 2)
        x = [conv_max_pool(x) for conv_max_pool in self.conv_max_pools]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# %% [markdown]
# ELMo

# %%
class ELMo(nn.Module):
    def __init__(
            self, 
            char_vocab: int, 
            char_embed_dim: int, 
            char_out_channels: list, 
            char_kernel_sizes: list, 
            dropout: float, 
            num_layers: int, 
            hidden_dim: int, 
            word_embed_dim: int,
            filename: str = None
        ) -> None:

        super(ELMo, self).__init__()

        self.char_cnn = CharCNN(
            char_vocab=char_vocab, 
            char_embed_dim=char_embed_dim, 
            char_out_channels=char_out_channels, 
            char_kernel_sizes=char_kernel_sizes, 
            dropout=dropout,
            word_embed_dim=word_embed_dim
        )

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

        if filename:
            self.load_state_dict(torch.load(filename))

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        bz = x.shape[0]
        x = x.view(-1, MAX_WORD_LEN)
        x = self.char_cnn(x)
        x = x.view(bz, -1, x.shape[1])
        xf = x
        xb = x.flip([1])
        input = x.detach().clone()
        xf, (hsf, csf) = self.lstmf(xf)
        xb, (hsb, csb) = self.lstmb(xb)
        xb = xb.flip([1])
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
            char_vocab: int,
            hidden_dim: int, 
            vocab_size: int, 
            filename: str = None
        ) -> None:

        super(LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.elmo = ELMo(
            char_vocab=char_vocab, 
            char_embed_dim=CHAR_EMBEDDING_DIM, 
            char_out_channels=[CHAR_OUT_CHANNELS] * 5,
            char_kernel_sizes=[2, 3, 4, 5, 6], 
            dropout=DROPOUT, 
            num_layers=NUM_LAYERS, 
            hidden_dim=HIDDEN_DIM,
            word_embed_dim=WORD_EMB_DIM
        )
        self.linear_forward = nn.Linear(hidden_dim, vocab_size)
        self.linear_backward = nn.Linear(hidden_dim, vocab_size)

        if filename:
            self.load_state_dict(torch.load(filename))

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        xf, xb, _, _, _ = self.elmo(x, l)
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
                torch.save(self.state_dict(), os.path.join(DIR, f'{filename}_lm.pth'))
                torch.save(self.elmo.state_dict(), os.path.join(DIR, f'{filename}_elmo.pth'))
                torch.save(self.elmo.char_cnn.state_dict(), os.path.join(DIR, f'{filename}_char_cnn.pth'))

    def _call(self, x: torch.Tensor, y: torch.Tensor, l: torch.Tensor, yf: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        x, y, yf, yb = x.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)
        # print(x.shape, y.shape, l.shape, yf.shape, yb.shape)
        yf_hat, yb_hat = self(x, l)

        yf_hat = yf_hat.view(-1, self.vocab_size)
        yb_hat = yb_hat.view(-1, self.vocab_size)

        yf = yf.view(-1)
        yb = yb.view(-1)

        # print(yf_hat.shape, yb_hat.shape, yf.shape, yb.shape)

        loss1 = self.criterion(yf_hat, yf)
        loss2 = self.criterion(yb_hat, yb)

        loss = (loss1 + loss2) / 2

        # print(loss1.item(), loss2.item(), loss.item())
        return loss, loss1, loss2

    def _train(self, train_loader: DataLoader) -> None:
        self.train()
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []

        pbar = tqdm(train_loader)
        for x, y, l, yf, yb in pbar:

            loss, loss1, loss2 = self._call(x, y, l, yf, yb)
            epoch_loss.append(loss.item())
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'T Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Avg Loss1: {np.mean(epoch_loss1):7.4f}, Avg Loss2: {np.mean(epoch_loss2):7.4f}')

        # run.log({'upstream_train_loss': np.mean(epoch_loss)})

    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.eval()
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []
        pbar = tqdm(validation_loader)
        with torch.no_grad():
            for x, y, l, yf, yb in pbar:
                loss, loss1, loss2 = self._call(x, y, l, yf, yb)
                epoch_loss.append(loss.item())
                epoch_loss1.append(loss1.item())
                epoch_loss2.append(loss2.item())
                pbar.set_description(f'V Loss: {epoch_loss[-1]:7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Avg Loss1: {np.mean(epoch_loss1):7.4f}, Avg Loss2: {np.mean(epoch_loss2):7.4f}, Counter: {self.es.counter}, Best Loss: {self.es.best_loss:7.4f}')

        # run.log({'upstream_validation_loss': np.mean(epoch_loss)})
        return np.mean(epoch_loss)

# %% [markdown]
# Initialize Model

# %%
lm = LM(char_vocab=len(char_to_idx), hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_idx)).to(DEVICE)
print(lm)

# %%
from torchinfo import summary

summary(lm, device=DEVICE)

# %%
lm.fit(dev_train_loader, dev_validation_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, filename='best')

# %%
downstream_train_raw_a = actual_df[DEV_TRAIN_LEN+DEV_VALIDATION_LEN:]
downstream_train_raw_p = pred_df[DEV_TRAIN_LEN+DEV_VALIDATION_LEN:]

downstream_validation_raw_a = actual_df[:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]
downstream_validation_raw_p = pred_df[:DEV_TRAIN_LEN+DEV_VALIDATION_LEN]

# %%
print(len(downstream_train_raw_a))
print(len(downstream_validation_raw_a))

# %%
downstream_train_dataset = Sentences(downstream_train_raw_a, downstream_train_raw_p, char_to_idx, word_to_idx)
downstream_validation_dataset = Sentences(downstream_validation_raw_a, downstream_validation_raw_p, char_to_idx, word_to_idx)

downstream_train_loader = DataLoader(downstream_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
downstream_validation_loader = DataLoader(downstream_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# %% [markdown]
# ## Downstream Task

# %%
from sklearn.metrics import classification_report, confusion_matrix

class NewsClassification(nn.Module):
    def __init__(self, 
            char_vocab: int,
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
            char_vocab=char_vocab, 
            char_embed_dim=CHAR_EMBEDDING_DIM, 
            char_out_channels=[CHAR_OUT_CHANNELS] * 5,
            char_kernel_sizes=[2, 3, 4, 5, 6], 
            dropout=DROPOUT, 
            num_layers=NUM_LAYERS, 
            hidden_dim=HIDDEN_DIM,
            word_embed_dim=WORD_EMB_DIM,
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

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        _, _, input, (hsf, csf), (hsb, csb) = self.elmo(x, l)
        hsf = hsf.permute(1, 0, 2)
        csf = csf.permute(1, 0, 2)
        hsb = hsb.permute(1, 0, 2)
        csb = csb.permute(1, 0, 2)

        hs = torch.cat([hsf, hsb], dim=2)
        cs = torch.cat([csf, csb], dim=2)

        val = (hs + cs) / 2

        input = torch.mean(input, dim=1).unsqueeze(1)

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

    def _call(self, x: torch.Tensor, y: torch.Tensor, l: torch.Tensor, yf: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        x, y, yf, yb = x.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)

        y_hat = self(x, l)
        y_hat = y_hat.view(-1, self.num_classes)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        return loss

    def _train(self, train_loader: DataLoader) -> None:
        self.train()
        epoch_loss = []
        pbar = tqdm(train_loader)
        for x, y, l, yf, yb in pbar:
            loss = self._call(x, y, l, yf, yb)
            epoch_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'T Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}')

    def _evaluate(self, validation_loader: DataLoader) -> float:
        self.eval()
        epoch_loss = []
        pbar = tqdm(validation_loader)
        with torch.no_grad():
            for x, y, l, yf, yb in pbar:
                loss = self._call(x, y, l, yf, yb)
                epoch_loss.append(loss.item())
                pbar.set_description(f'V Loss: {epoch_loss[-1]:7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Counter: {self.es.counter}, Best Loss: {self.es.best_loss:7.4f}')
        return np.mean(epoch_loss)

    def _metrics(self, test_loader: DataLoader) -> None:
        self.eval()
        self.criterion = nn.CrossEntropyLoss()
        pbar = tqdm(test_loader)
        y_pred = []
        y_true = []
        epoch_loss = []

        with torch.no_grad():
            for x, y, l, yf, yb in pbar:
                x, y, yf, yb = x.to(DEVICE), y.to(DEVICE), yf.to(DEVICE), yb.to(DEVICE)
                y_hat = self(x, l)
                y_hat = y_hat.view(-1, self.num_classes)
                y = y.view(-1)
                loss = self.criterion(y_hat, y)

                epoch_loss.append(loss.item())
                y_hat = torch.argmax(y_hat, dim=1)
                y_pred += y_hat.tolist()
                y_true += y.tolist()

        print(f'Test Loss: {np.mean(epoch_loss):7.4f}')

        cr = classification_report(y_true, y_pred, digits=4)
        print('Classification Report:', cr)

        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:', cm)

# %%
nc = NewsClassification(char_vocab=len(char_to_idx), 
                        hidden_dim=HIDDEN_DIM, 
                        vocab_size=len(word_to_idx), 
                        num_classes=NUM_CLASSES,
                        filename=os.path.join(DIR, 'best_elmo.pth')
                    ).to(DEVICE)

# %%
summary(nc, device=DEVICE)

# %%
nc.fit(downstream_train_loader, downstream_validation_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# %%
nc.load_state_dict(torch.load(os.path.join(DIR, 'best.pth')))

# %%
test_adf, test_pdf = read_data('data/test.csv', create_unique_words=False, filter_rare_words=False)

# %%
downstream_test_dataset = Sentences(test_adf, test_pdf, char_to_idx, word_to_idx)
downstream_test_loader = DataLoader(downstream_test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# %%
nc._metrics(downstream_test_loader)

# %%



