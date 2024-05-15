import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
from torchinfo import summary

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim=18432, feedforward_dim=512, output_dim=5, batch_first=True, num_heads=4, num_layers=4):
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEncoding(input_dim)
        self.transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=feedforward_dim)

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # Add positional encoding to the input
        src = self.positional_encoding(src)
        
        output = self.transformer(src, src)  # Transformer forward pass
        output = self.fc(output[-1])  # Take the output from the last time step and pass through linear layer
        return output
