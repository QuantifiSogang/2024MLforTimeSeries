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

class GRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_ih, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=np.sqrt(5))
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, input, hidden):
        h_next = torch.tanh(torch.matmul(input, self.W_ih.t()) + self.b_ih +
                            torch.matmul(hidden, self.W_hh.t()) + self.b_hh)
        return h_next