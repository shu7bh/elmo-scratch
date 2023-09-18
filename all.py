# %%
cfg = {
    'dev_train_len': 25*10**3,
    'dev_validation_len': 5*10**3,
    'learning_rate': 0.001,
    'epochs': 100,
    'embedding_dim': 50,
    # 'hidden_dim': 50,
    'dropout': 0.1,
    'optimizer': 'Adam',
    'num_layers': 2
}

cfg['hidden_dim'] = cfg['embedding_dim']

# %%
from nltk import word_tokenize
from gensim.models import KeyedVectors
import gensim.downloader as api
import unicodedata
import random
import torch
import re

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)

def tokenize_corpus(s: str) -> str:
    s = normalize_unicode(s)
    s = s.lower()
    s = re.sub(r"""[^a-zA-Z0-9?.,;'"]+""", " ", s)
    s = re.sub(r'(.)\1{3,}',r'\1', s)
    s = s.rstrip().strip()
    return s

def get_word_tokenized_corpus(s: str) -> list:
    return word_tokenize(s)

glove_dict = {
    '50': 'glove-wiki-gigaword-50',
    '100': 'glove-wiki-gigaword-100',
    '200': 'glove-wiki-gigaword-200'
}

glove_dict['50'] = api.load(glove_dict['50'])
# glove_dict['100'] = api.load(glove_dict['100'])
# glove_dict['200'] = api.load(glove_dict['200'])

def create_vocab(sentences: list, embedding_dim: int):
    glove = glove_dict[str(embedding_dim)]

    Emb = KeyedVectors(vector_size=glove.vector_size)
    vocab = []

    for sentence in sentences:
        vocab.extend(sentence)

    vocab = set(vocab)

    vectors, keys = [], []
    for token in vocab:
        if token in glove:
            vectors.append(torch.tensor(glove[token]))
            keys.append(token)

    keys.extend(['<unk>', '<pad>', '<sos>', '<eos>'])
    vectors.append(torch.mean(torch.stack(vectors), dim=0).numpy())
    vectors.append([0 for _ in range(embedding_dim)])
    vectors.append([random.random() for _ in range(embedding_dim)])
    vectors.append([random.random() for _ in range(embedding_dim)])
    Emb.add_vectors(keys, vectors)

    return Emb

def get_sentence_index(sentence: list, Emb: KeyedVectors):
    word_vec = []

    word_vec.append(Emb.key_to_index['<sos>'])
    for word in sentence:
        word_vec.append(get_vocab_index(word, Emb))
    word_vec.append(Emb.key_to_index['<eos>'])

    return torch.tensor(word_vec)

def get_vocab_index(word: str, Emb: KeyedVectors):
    if word in Emb:
        return Emb.key_to_index[word]
    return Emb.key_to_index['<unk>']


# %%
import numpy as np
import torch
import pandas as pd

# %%
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# %%
# DEV_TRAIN_LEN = cfg['parameters']['dev_train_len']['value']
# DEV_VALIDATION_LEN = cfg['parameters']['dev_validation_len']['value']

DEV_TRAIN_LEN = cfg['dev_train_len']
DEV_VALIDATION_LEN = cfg['dev_validation_len']

DIR = '/scratch/shu7bh/RES/PRE'

# %%
import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

# %%
print(DEV_TRAIN_LEN)
print(DEV_VALIDATION_LEN)

# %%
df = pd.read_csv('data/train.csv')
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
df['Description'] = df['Description'].apply(tokenize_corpus)
df['Description'] = df['Description'].apply(get_word_tokenized_corpus)

# %%
df['Class Index']

# %%
dev_train = df[:DEV_TRAIN_LEN]['Description']
dev_validation = df[DEV_TRAIN_LEN:DEV_TRAIN_LEN + DEV_VALIDATION_LEN]['Description']

# %%
dev_validation

# %%
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# %%
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ELMO(nn.Module):
    def __init__(self, Emb, hidden_dim, dropout, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(Emb.vectors), padding_idx=Emb.key_to_index['<pad>'])
        self.lstm = nn.LSTM(Emb.vectors.shape[1], hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)

    def forward(self, X, X_lengths):
        X = self.embedding(X)
        X = pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        # X, _ = self.lstm(X, None)
        X, (h_n, c_n) = self.lstm(X, None)
        X, _ = pad_packed_sequence(X, batch_first=True)
        return X, h_n, c_n

# %%
from torch import nn

class LM(nn.Module):
    def __init__(self, Emb, hidden_dim, dropout, num_layers):
        super().__init__()

        self.elmo = ELMO(Emb, hidden_dim, dropout, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, Emb.vectors.shape[0])

    def forward(self, X, X_lengths):
        X, _, _ = self.elmo(X, X_lengths)
        X = self.fc(X)
        return X

# %%
from torch.utils.data import Dataset
import torch

class SentencesDataset(Dataset):
    def __init__(self, sentences: list, Emb):
        super().__init__()

        self.data = []
        for sentence in sentences:
            self.data.append(get_sentence_index(sentence, Emb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(len(self.data[idx]))

# %%
class Collator:
    def __init__(self, Emb):
        self.pad_index = Emb.key_to_index['<pad>']

    def __call__(self, batch):
        X, X_lengths = zip(*batch)
        X = pad_sequence(X, batch_first=True, padding_value=self.pad_index)
        return X[:, :-1], X[:, 1:], torch.stack(X_lengths) - 1

# %%
import tqdm

def fit(model, dataloader, train, es, loss_fn, optimizer):
    model.train() if train else model.eval()
    epoch_loss = []

    pbar = tqdm.tqdm(dataloader)

    for X, Y, X_lengths in pbar:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        Y_pred = model(X, X_lengths)
        Y_pred = Y_pred.reshape(-1, Y_pred.shape[2])

        Y = Y.reshape(-1)

        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        X.detach()
        Y_pred.detach()
        Y.detach()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f'{"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Best Loss: {es.best_loss:7.4f}, Counter: {es.counter}')

    return np.mean(epoch_loss)

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

# %%
def train(EPOCHS, model, training_dataloader, validation_dataloader, loss_fn, optimizer):
    es = EarlyStopping(patience=3, delta=0.001)

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}')

        epoch_loss = fit(model, training_dataloader, True, es, loss_fn, optimizer)
        # wandb.log({'train_loss': epoch_loss})

        with torch.no_grad():
            epoch_loss = fit(model, validation_dataloader, False, es, loss_fn, optimizer)
            # wandb.log({'validation_loss': epoch_loss})
            if es(epoch_loss, epoch):
                break
            if es.counter == 0:
                torch.save(model.state_dict(), os.path.join(DIR, f'best_model.pth'))
                torch.save(model.elmo.state_dict(), os.path.join(DIR, f'best_model_elmo.pth'))

# %%
def run(config=None):
    # with wandb.init(config=cfg):
        # config = wandb.config
    BATCH_SIZE = 16
    # if config.hidden_dim in [300, 500]:
    #     BATCH_SIZE = 32
    if config['hidden_dim'] in [300, 500]:
        BATCH_SIZE = 32

    # wandb.log({'batch_size': BATCH_SIZE})
    HIDDEN_DIM = config['hidden_dim']
    DROP_OUT = config['dropout']
    OPTIMIZER = config['optimizer']
    LEARNING_RATE = config['learning_rate']
    EPOCHS = config['epochs']
    EMBEDDITNG_DIM = config['embedding_dim']
    NUM_LAYERS = config['num_layers']

    Emb = create_vocab(df['Description'], EMBEDDITNG_DIM)

    dev_train_dataset = SentencesDataset(dev_train, Emb)
    dev_validation_dataset = SentencesDataset(dev_validation, Emb)

    collate_fn = Collator(Emb)

    training_dataloader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    validation_dataloader = DataLoader(dev_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    torch.cuda.empty_cache()

    model = LM(Emb, HIDDEN_DIM, DROP_OUT, NUM_LAYERS).to(DEVICE)

    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=Emb.key_to_index['<pad>'])

    train(EPOCHS, model, training_dataloader, validation_dataloader, loss_fn, optimizer)

run(cfg)

# %%
import tqdm

def run_epoch(model, dataloader, loss_fn):
    epoch_loss = []

    pbar = tqdm.tqdm(dataloader)

    for X, Y, X_lengths in pbar:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        Y_pred = model(X, X_lengths)
        Y_pred = Y_pred.reshape(-1, Y_pred.shape[2])

        Y = Y.reshape(-1)

        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        pbar.set_description(f'Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}')

    return np.mean(epoch_loss)

# %%
def validate(elmo, validation_dataloader, loss_fn):
    with torch.no_grad():
        elmo.eval()
        epoch_loss = run_epoch(elmo, validation_dataloader, loss_fn)
        print(f'Validation Loss: {epoch_loss:7.4f}')

# %%
Emb = create_vocab(df['Description'], cfg['embedding_dim'])

# dev_train_dataset = SentencesDataset(dev_train, Emb)
dev_validation_dataset = SentencesDataset(dev_validation, Emb)

collate_fn = Collator(Emb)

# training_dataloader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
validation_dataloader = DataLoader(dev_validation_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
model = LM(Emb, cfg['hidden_dim'], cfg['dropout'], cfg['num_layers']).to(DEVICE)

model.load_state_dict(torch.load(os.path.join(DIR, 'best_model.pth')))

validate(model, validation_dataloader, nn.CrossEntropyLoss(ignore_index=Emb.key_to_index['<pad>']))

# %%
elmo = ELMO(Emb, cfg['hidden_dim'], cfg['dropout'], cfg['num_layers']).to(DEVICE)

# load only elmo from the best model
elmo.load_state_dict(torch.load(os.path.join(DIR, 'best_model_elmo.pth')), strict=False)

# %%
from torch.utils.data import DataLoader, Dataset

class DownStreamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, Emb: KeyedVectors):
        self.descriptions = df['Description']
        self.descriptions = [get_sentence_index(description, Emb) for description in self.descriptions]
        # self.df.loc[: 'Description'] = df['Description'].apply(get_sentence_index, Emb=Emb)
        self.class_index = list(c - 1 for c in df['Class Index'])
        self.Emb = Emb

    def __len__(self):
        return len(self.class_index)

    def __getitem__(self, idx):
        l = torch.tensor(len(self.descriptions[idx]))
        return self.descriptions[idx], l, torch.tensor(self.class_index[idx])

# %%
class DownStreamCollator:
    def __init__(self, Emb):
        self.pad_index = Emb.key_to_index['<pad>']

    def __call__(self, batch):
        X, X_lengths, Y = zip(*batch)
        X = pad_sequence(X, batch_first=True, padding_value=self.pad_index)
        # print(len(X_lengths))
        # print(len(Y))
        # print('hello')
        return X, torch.stack(X_lengths), torch.stack(Y)

# %%
downstream_train = df[DEV_TRAIN_LEN + DEV_VALIDATION_LEN:]
downstream_validation = df[:DEV_TRAIN_LEN + DEV_VALIDATION_LEN]

downstream_train_dataset = DownStreamDataset(downstream_train, Emb)
downstream_validation_dataset = DownStreamDataset(downstream_validation, Emb)

downstream_collate_fn = DownStreamCollator(Emb)

downstream_training_dataloader = DataLoader(downstream_train_dataset, batch_size=64, shuffle=True, collate_fn=downstream_collate_fn, pin_memory=True)
downstream_validation_dataloader = DataLoader(downstream_validation_dataset, batch_size=64, shuffle=True, collate_fn=downstream_collate_fn, pin_memory=True)

# %%
class DownStream(nn.Module):
    def __init__(self, elmo, dropout):
        super().__init__()
        self.elmo = elmo
        # freeze the ELMO parameters
        for param in self.elmo.parameters():
            param.requires_grad = False

        self.hidden_dim = self.elmo.hidden_dim
        self.num_layers = self.elmo.num_layers
        self.dropout = dropout

        self.delta = nn.Parameter(torch.randn(1, self.num_layers + 1))
        self.linear = nn.Linear(self.hidden_dim * 2, 4)

    def forward(self, X, X_lengths):
        _, Y1, Y2 = self.elmo(X, X_lengths)
        # get the first hidden layer batch_first=True

        Y = torch.mean(torch.stack([Y1, Y2]), dim=0)

        # print(Y.shape)
        Y = Y.permute(1, 0, 2)
        
        # print(Y.shape)
        
        Y = Y.reshape(Y.shape[0], self.num_layers, self.hidden_dim * 2)

        X = self.elmo.embedding(X)
        X = torch.mean(X, dim=1, dtype=torch.float32)
        X = torch.cat([X, X], dim=1)
        X = X.unsqueeze(1)

        Y = torch.cat([X, Y], dim=1)

        # multiply by delta
        Y = (self.delta / torch.sum(self.delta) ) @ Y

        # sum over the num_layers dimension
        Y = torch.sum(Y, dim=1)

        # pass through linear layer
        Y = self.linear(Y)

        return Y

# %%
def downstream_fit(model, dataloader, train, loss_fn, optimizer):
    model.train() if train else model.eval()
    epoch_loss = []

    pbar = tqdm.tqdm(dataloader)

    for X, X_lengths, Y in pbar:
        # print('hello')
        # print(X.shape)
        # print(X_lengths.shape)
        # print(Y.shape)
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        Y_pred = model(X, X_lengths)
        Y_pred = Y_pred.reshape(-1, Y_pred.shape[-1])

        # Y_pred = Y_pred.softmax(dim=1)
        # Y_pred = torch.argmax(Y_pred, dim=1)

        # Y = Y.reshape(-1)

        # print(Y_pred == Y)

        # break

        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        X.detach()
        Y_pred.detach()
        Y.detach()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f'{"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}')

    return np.mean(epoch_loss)

# %%
dmodel = DownStream(elmo, cfg['dropout']).to(DEVICE)
doptimizer = getattr(torch.optim, cfg['optimizer'])(dmodel.parameters(), lr=cfg['learning_rate'])
dloss_fn = nn.CrossEntropyLoss()

def downstream_train_fn(EPOCHS, model, training_dataloader, validation_dataloader, loss_fn, optimizer):
    es = EarlyStopping(patience=3, delta=0.001)
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}')

        epoch_loss = downstream_fit(model, training_dataloader, True, loss_fn, optimizer)
        print(f'Train Loss: {epoch_loss:7.4f}')
        # wandb.log({'downstream_train_loss': epoch_loss})

        with torch.no_grad():
            epoch_loss = downstream_fit(model, validation_dataloader, False, loss_fn, optimizer)
            print(f'Validation Loss: {epoch_loss:7.4f}')
            if es(epoch_loss, epoch):
                break
            if es.counter == 0:
                torch.save(model.state_dict(), os.path.join(DIR, f'downstream_best_model.pth'))
            # wandb.log({'downstream_validation_loss': epoch_loss})

# %%
downstream_train_fn(cfg['epochs'], dmodel, downstream_training_dataloader, downstream_validation_dataloader, dloss_fn, doptimizer)

# %%
tdf = pd.read_csv('data/test.csv')
tdf = tdf.sample(frac=1, random_state=0).reset_index(drop=True)
tdf['Description'] = tdf['Description'].apply(tokenize_corpus)
tdf['Description'] = tdf['Description'].apply(get_word_tokenized_corpus)

# %%
downstream_train_dataset = DownStreamDataset(tdf, Emb)
downstream_test_dataloader = DataLoader(downstream_train_dataset, batch_size=32, shuffle=True, collate_fn=downstream_collate_fn, pin_memory=True, num_workers=4)

# %%
dmodel.load_state_dict(torch.load(os.path.join(DIR, 'downstream_best_model.pth')))

with torch.no_grad():
    print(f'Test loss: {downstream_fit(dmodel, downstream_test_dataloader, False, dloss_fn, doptimizer)}')