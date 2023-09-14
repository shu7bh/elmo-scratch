# %%
from config import config as cfg
from preprocess import *
import numpy as np
import wandb
import torch
import pandas as pd
import os

# %%
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# %%
DEV_TRAIN_LEN = cfg['parameters']['dev_train_len']['value']
DEV_VALIDATION_LEN = cfg['parameters']['dev_validation_len']['value']

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
dev_train = df[:DEV_TRAIN_LEN]['Description']
dev_validation = df[DEV_TRAIN_LEN:DEV_TRAIN_LEN + DEV_VALIDATION_LEN]['Description']

# %%
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset import SentencesDataset
from elmo import ELMO
from torch import nn

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

def run(model, dataloader, train, es, loss_fn, optimizer):
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
import EarlyStopping as ES

def train(EPOCHS, elmo, training_dataloader, validation_dataloader, loss_fn, optimizer):
    es = ES.EarlyStopping()

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}')

        epoch_loss = run(elmo, training_dataloader, True, es, loss_fn, optimizer)
        wandb.log({'train_loss': epoch_loss})

        with torch.no_grad():
            epoch_loss = run(elmo, validation_dataloader, False, es, loss_fn, optimizer)
            wandb.log({'validation_loss': epoch_loss})
            if es(epoch_loss, epoch):
                break
    
        torch.save(elmo.state_dict(), os.path.join(DIR, f'elmo_{epoch + 1}.pth'))

    wandb.log({'loss': es.best_loss})
    os.rename(os.path.join(DIR, f'elmo_{es.best_model_pth + 1}.pth'), os.path.join(DIR, 'best_model.pth'))

    wandb.save(os.path.join(DIR, 'best_model.pth'))

# %%
def run_sweep(cfg=None):
    with wandb.init(config=cfg):
        config = wandb.config
        BATCH_SIZE = 16
        if config.hidden_dim in [300, 500]:
            BATCH_SIZE = 32
        # else:
            # config.batch_size = 16
        # BATCH_SIZE = config['batch_size']

        wandb.log({'batch_size': BATCH_SIZE})
        HIDDEN_DIM = config['hidden_dim']
        DROP_OUT = config['dropout']
        OPTIMIZER = config['optimizer']
        LEARNING_RATE = config['learning_rate']
        EPOCHS = config['epochs']
        EMBEDDITNG_DIM = config['embedding_dim']
        NUM_LAYERS = config['num_layers']

        # wandb.log({'loss': np.random.random()})
        # return

        Emb = create_vocab(df['Description'], EMBEDDITNG_DIM)

        dev_train_dataset = SentencesDataset(dev_train, Emb)
        dev_validation_dataset = SentencesDataset(dev_validation, Emb)

        collate_fn = Collator(Emb)

        training_dataloader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
        validation_dataloader = DataLoader(dev_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)

        torch.cuda.empty_cache()

        elmo = ELMO(Emb, HIDDEN_DIM, DROP_OUT, NUM_LAYERS).to(DEVICE)

        wandb.watch(elmo, log_freq=100)

        optimizer = getattr(torch.optim, OPTIMIZER)(elmo.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss(ignore_index=Emb.key_to_index['<pad>'])

        train(EPOCHS, elmo, training_dataloader, validation_dataloader, loss_fn, optimizer)

sweep_id = wandb.sweep(cfg, project='ELMO')
wandb.agent(sweep_id, run_sweep, count=20)

# %%



