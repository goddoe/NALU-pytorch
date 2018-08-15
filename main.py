import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nalu import NALU
from nac import NAC
from datasets import ArithmeticDataset
import datasets as d


def train(logic_unit_class=NALU,
          func_type=d.ADD,
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
    logic_unit = logic_unit_class(input_dim=2,
                                  output_dim=1)

    # Training Parameters
    optimizer = optim.Adam(logic_unit.parameters(),
                           lr=learning_rate)
    loss_fn = nn.MSELoss()
    loss_list = []

    # Use GPU, if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logic_unit.to(device)

    for epoch_i in range(n_epoch):
        for batch_i, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            logic_unit.zero_grad()
            pred_log_prob = logic_unit(X)

            loss = loss_fn(pred_log_prob, Y.view((-1, 1)))

            loss.backward()
            loss_list.append(float(loss.to('cpu').data.numpy()))

            optimizer.step()

            if epoch_i % verbose_iterval == 0:
                print("epoch_i: {}, loss : {:.3f}".format(epoch_i, loss_list[-1]))

    return {'ad': ad,
            'logic_unit': logic_unit,
            'loss_list': loss_list,
            'data_loader': data_loader}


if __name__ == '__main__':

    func_type = d.ADD
    data_size = 1000
    low = 0
    high = 10

    result = train(logic_unit_class=NALU,
                   func_type=func_type,
                   data_size=data_size,
                   low=low,
                   high=high,
                   n_epoch=500,
                   batch_size=32,
                   learning_rate=0.01,
                   shuffle=True,
                   verbose_iterval=1)

    def _exec_logic_unit(*args):
        if len(args) == 2:
            return result['logic_unit'](
                    torch.Tensor([[args[0], args[1]]])).data.numpy()
        elif len(args) == 1 and type(args[0]) == torch.Tensor:
            return result['logic_unit'](
                    args[0].view((1, -1))).data.numpy()
        else:
            raise Exception("args should be either two number or a tensor")

    def _evaluate(desc, func_type, data_size, low, high, decimal=1):
        ad = ArithmeticDataset(func_type=func_type,
                               data_size=data_size,
                               low=low,
                               high=high)
        total = 0
        success = 0
        fail_case_list = []
        for i, (X, Y) in enumerate(ad):
            try:
                total += 1
                print(f"{desc}, {i}th", end="...")
                print(f"{func_type}, X: {X.tolist()}, Y: {Y.tolist()}")
                np.testing.assert_array_almost_equal(Y.numpy(),
                                                     _exec_logic_unit(X),
                                                     decimal=decimal)
                print("pass")
                success += 1
            except Exception as e:
                print("")
                print("*"*30)
                print(e)
                print(X.tolist(), Y.tolist())
                print("*"*30)
                fail_case_list.append((X.tolist(), Y.tolist()))

        return total, success, fail_case_list

    # Interpolation
    (total_inter,
     success_inter,
     fail_case_list) = _evaluate(desc="Interpolation",
                                 func_type=func_type,
                                 data_size=data_size,
                                 low=low,
                                 high=high)

    # Extrapolation
    data_size = 10000
    low = 0
    high = 100

    (total_extra,
     success_extra,
     fail_case_list) = _evaluate(desc="Extrapolation",
                                 func_type=func_type,
                                 data_size=data_size,
                                 low=low,
                                 high=high)

    print(f"total: {total_inter}, success: {success_inter}, fail: {total_inter-success_inter}")
    print(f"total: {total_extra}, success: {success_extra}, fail: {total_extra-success_extra}")
