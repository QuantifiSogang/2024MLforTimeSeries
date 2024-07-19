import numpy as np
import torch
import torch.nn as nn
from TimeML.torch_extension.RNNCell import *

class AlphaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AlphaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [AlphaRNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, input, hidden=None):
        batch_size, seq_len, _ = input.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x, hidden[i])
                x = hidden[i]
            outputs.append(x.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].W_ih.device) for _ in
                range(self.num_layers)]

class AlphatRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AlphatRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([AlphatRNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, input, hx=None):
        if hx is None:
            hx = [torch.zeros(input.size(0), self.hidden_size, device=input.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(input.size(1)):
            x = input[:, t, :]
            for i, cell in enumerate(self.cells):
                hx[i], _ = cell(x, (hx[i],))
                x = hx[i]
            outputs.append(x.unsqueeze(1))

        return torch.cat(outputs, dim=1), hx

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].W_ih.device) for _ in range(self.num_layers)]

class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [GRNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, input, hidden=None):
        batch_size, seq_len, _ = input.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x, hidden[i])
                x = hidden[i]
            outputs.append(x.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].W_ih.device) for _ in range(self.num_layers)]