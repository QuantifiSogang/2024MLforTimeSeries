import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AlphaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.alpha = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_ih, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=np.sqrt(5))
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)
        nn.init.constant_(self.alpha, 0.5)

    def forward(self, input, hidden):
        h_next = torch.sigmoid(self.alpha) * hidden + (1 - torch.sigmoid(self.alpha)) * torch.tanh(
            F.linear(input, self.W_ih, self.b_ih) + F.linear(hidden, self.W_hh, self.b_hh)
        )
        return h_next

class AlphatRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True):
        super(AlphatRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias

        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if use_bias:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_ih, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=np.sqrt(5))
        if self.b_ih is not None:
            nn.init.zeros_(self.b_ih)
            nn.init.zeros_(self.b_hh)
        nn.init.constant_(self.alpha, 0.5)

    def forward(self, input, hx):
        h_prev = hx[0]
        Wx = F.linear(input, self.W_ih, self.b_ih)
        Wh = F.linear(h_prev, self.W_hh, self.b_hh)
        alpha = torch.sigmoid(Wx + Wh)
        if self.activation == 'tanh':
            activation_fn = torch.tanh
        elif self.activation == 'relu':
            activation_fn = torch.relu
        else:
            activation_fn = lambda x: x
        h_new = alpha * h_prev + (1 - alpha) * activation_fn(Wx + Wh)
        return h_new, h_new.unsqueeze(0)

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