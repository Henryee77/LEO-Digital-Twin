"""autoencoder.py"""
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Model structure


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(784, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 2),
        nn.Tanh()
    )

  def forward(self, inputs):
    codes = self.encoder(inputs)
    return codes


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 128),
        nn.Tanh(),
        nn.Linear(128, 256),
        nn.Tanh(),
        nn.Linear(256, 784),
        nn.Sigmoid()
    )

  def forward(self, inputs):
    outputs = self.decoder(inputs)
    return outputs


class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, inputs):
    codes = self.encoder(inputs)
    decoded = self.decoder(codes)
    return codes, decoded
