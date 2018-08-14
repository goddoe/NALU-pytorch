import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAC(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(NAC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_hat = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.M_hat = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.W = nn.Parameter(F.tanh(self.W_hat)*F.sigmoid(self.M_hat))

        nn.init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))

    def forward(self, X):
        return X.mm(self.W)
