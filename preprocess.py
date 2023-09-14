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
glove_dict['100'] = api.load(glove_dict['100'])
glove_dict['200'] = api.load(glove_dict['200'])

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