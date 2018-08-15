import numpy as np
import numpy.random as random
import torch
from torch.utils.data import Dataset

ADD = 'ADD'
SUB = 'SUB'
MUL = 'MUL'
DIV = 'DIV'


def _to_tensor(sample):
    return (torch.tensor(sample[0], dtype=torch.float, requires_grad=False),
            torch.tensor(sample[1], dtype=torch.float, requires_grad=False))


def _gen_add_data(data_size, low=0, high=30):
    X = random.randint(low=low,
                       high=high,
                       size=data_size*2).reshape((-1, 2))
    Y = np.sum(X, axis=1)
    return X, Y


def _gen_sub_data(data_size, low=0, high=30):
    X = random.randint(low=low,
                       high=high,
                       size=data_size*2).reshape((-1, 2))
    Y = X[:, 0] - X[:, 1]
    return X, Y


def _gen_mul_data(data_size, low=0, high=30):
    X = random.randint(low=low,
                       high=high,
                       size=data_size*2).reshape((-1, 2))
    Y = X[:, 0] * X[:, 1]
    return X, Y


def _gen_div_data(data_size, low=0, high=30):
    X = random.randint(low=low,
                       high=high,
                       size=data_size*2).reshape((-1, 2))
    X[X == 0] = 1
    Y = X[:, 0] / X[:, 1]
    return X, Y


_func_dict = {ADD: _gen_add_data,
              SUB: _gen_add_data,
              MUL: _gen_mul_data,
              DIV: _gen_div_data}


class ArithmeticDataset(Dataset):
    """

    Attributes:
        pass
    """

    def __init__(self, func_type, data_size, low=0, high=30):
        """
        Args:
            func_type (str): A string from 'ADD', 'SUB', 'MUL', 'DIV',
                            A specifier of arithmetic fuctions.
            data_size (int): A size of data.
        """
        global _func_dict
        func = _func_dict[func_type]

        self.X, self.Y = func(data_size=data_size,
                              low=low,
                              high=high)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = self.X[idx], self.Y[idx]
        return _to_tensor(sample)


