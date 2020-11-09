import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import pdb

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        return outputs, hidden,embedded



class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        return outputs, hidden

class Latent_Discriminator(nn.Module):
    def __init__(self, Layer1, Layer2):
        super().__init__()
        self.main_module = nn.Sequential(
            torch.nn.Linear(Layer1, Layer2),
            torch.nn.ReLU(),
            torch.nn.Linear(Layer2,2),
        )
    def forward(self, x):
        x = self.main_module(x)
        return x


class RVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(RVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, src):
        encoder_output, hidden_enc,embedded = self.encoder(src)
        output, hidden_dec = self.decoder(encoder_output)
        #output, hidden = self.decoder(encoder_output, hidden)
        return output,embedded, encoder_output


