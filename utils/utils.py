import re
import spacy
import numpy as np
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import torch.nn.functional as F
import pdb
def one_hot_vetorisation(seq_batch,one_hot_dim):
    _shape = np.shape(seq_batch)
    _one_hot = np.zeros([_shape[0],_shape[1],one_hot_dim])
    for _idx,_tmp in enumerate(seq_batch):
        _one_hot[_idx] = F.one_hot(_tmp,num_classes=one_hot_dim)
    return _one_hot


def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN