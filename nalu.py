import math
import torch
import torch.nn as nn
from nac import NAC


class NALU(nn.Module):

    def __init__(self, input_dim, output_dim, eps=1e-8):
        super(NALU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps

        self.nac = NAC(input_dim, output_dim)
        self.G = nn.Parameter(torch.Tensor(input_dim, output_dim))

        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, X):
        gate = torch.sigmoid(X.mm(self.G))
        add_sub = self.nac(X) * gate
        mul_div = torch.exp(
                    self.nac(
                        torch.log(torch.abs(X)+self.eps))) * (1.-gate)
        return add_sub+mul_div
