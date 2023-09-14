from preprocess import get_sentence_index
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