import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nalu import NALU
from datasets import (ArithmeticDataset,
                      ADD,
                      SUB,
                      MUL,
                      DIV)


def train(func_type=ADD,
          data_size=1000,
          low=0,
          high=50,
          n_epoch=10,
          batch_size=32,
          learning_rate=0.0001,
          shuffle=True,
          verbose_iterval=1):

    # Load data
    ad = ArithmeticDataset(func_type=func_type,
                           data_size=data_size,
                           low=low,
                           high=high)

    data_loader = DataLoader(ad,
                             batch_size=batch_size,
                             shuffle=shuffle)

    # Model
    nalu = NALU(input_dim=2,
                output_dim=1)

    # Training Parameters
    optimizer = optim.SGD(nalu.parameters(),
                          lr=learning_rate)
    loss_fn = nn.MSELoss()
    loss_list = []

    # Use GPU, if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nalu.to(device)

    for epoch_i in range(n_epoch):
        for batch_i, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            nalu.zero_grad()
            pred_log_prob = nalu(X).view((-1))

            loss = loss_fn(pred_log_prob, Y)

            loss.backward()
            loss_list.append(float(loss.to('cpu').data.numpy()))

            optimizer.step()

            if epoch_i % verbose_iterval == 0:
                print("epoch_i: {}, loss : {:.3f}".format(epoch_i, loss_list[-1]))

    return {'ad': ad,
            'nalu': nalu,
            'loss_list': loss_list,
            'data_loader': data_loader}



if __name__ == '__main__':
    result = train(func_type=ADD,
                   data_size=1000,
                   low=0,
                   high=30,
                   n_epoch=100,
                   batch_size=32,
                   learning_rate=0.0001,
                   shuffle=True,
                   verbose_iterval=1)

    def _exec_nalu(a, b):
        return result['nalu'](torch.Tensor([[a, b]]))

    np.testing.assert_array_almost_equal(3, _exec_nalu(1,2).to('cpu').data.numpy(), decimal=2)




