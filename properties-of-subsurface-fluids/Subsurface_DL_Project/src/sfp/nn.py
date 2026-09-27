"""The network, the training loop and the physics term.

Everything here follows the supplied Deep Learning notebooks:

* ``Day01-Intro_DL_FFNs_and_Colab_morning_solutions.ipynb``
  cell 21  ``set_seed`` (all seeds fixed, cuDNN autotuner off)
  cell 23  ``class simpleFFN(nn.Module)`` -- ``nn.Linear`` stack built with
           ``setattr`` in ``__init__`` and read back with ``getattr`` in
           ``forward``, one activation applied after every layer including
           the output layer
  cell 35  ``train(model, optimizer, criterion, data_loader)`` and
           ``validate(...)``, loss accumulated as ``loss * X.size(0)`` and
           divided by ``len(data_loader.dataset)``
  cell 41  ``train_loop`` -- seed, optimiser, criterion, DataLoaders, epochs
* ``Day02-Intro_to_Pytorch_afternoon_solutions.ipynb`` /
  ``Day03-Intro_to_Pytorch_morning_solutions.ipynb``
  ``nn.MSELoss``, ``torch.optim.Adam``, ``TensorDataset`` + ``DataLoader``,
  ``torch.save(model.state_dict(), ...)``
* ``Day04-CNNs(1)_morning_solutions.ipynb``  ``nn.ReLU``

The physics term follows the physics-informed loss of
``Day13-SciML_morning.pdf`` slide 37:

    L(theta) = (1/N) SUM [ NN(x_i; theta) - u_i ]^2
             + (lambda/M) SUM [ D( NN(x_j; theta) ) ]^2

where ``D`` is the governing equation the solution must satisfy.  Here ``D``
is the Rachford-Rice function of ``Models/3 - Two-phase flash calculation.pdf``
p. 8, evaluated at the network's own prediction:

    h(F_V) = SUM_i z_i (K_i - 1) / [ F_V (K_i - 1) + 1 ]

so the second term asks the prediction to satisfy the same equation the
bisection solver solves.  It uses no label.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    """Day01 solutions, cell 21."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True


class simpleFFN(nn.Module):
    """Feed-forward network, Day01 solutions cell 23, adapted to regression.

    The only changes to the taught class are the ones the task requires: a
    single output instead of ten, biases on, ``nn.ReLU`` between hidden layers
    and ``nn.Sigmoid`` on the output because the target is a vapour fraction
    and ``0 < F_V < 1`` by definition (flash notes p. 2, F_V + F_L = 1).
    """

    def __init__(self, n_inputs, num_hidden=(64, 64), bias=True,
                 activation=nn.ReLU, output_activation=nn.Sigmoid):
        super(simpleFFN, self).__init__()
        hidden_input = n_inputs
        self.len_hidden = len(num_hidden)
        for idx, n in enumerate(num_hidden):
            setattr(self, "hidden_%d" % idx, nn.Linear(hidden_input, n, bias=bias))
            hidden_input = n
        self.output = nn.Linear(hidden_input, 1, bias=bias)
        self.activation = activation()
        self.output_activation = output_activation()

    def forward(self, x):
        for idx in range(self.len_hidden):
            hidden = getattr(self, "hidden_%d" % idx)
            x = self.activation(hidden(x))
        return self.output_activation(self.output(x)).squeeze(-1)


def rachford_rice_torch(FV, z, K):
    """h(F_V) in torch -- flash notes p. 8.  ``FV`` is ``(n,)``, ``z``/``K``
    are ``(n, n_c)``.  Differentiable in ``FV``."""
    Km1 = K - 1.0
    return (z * Km1 / (FV.unsqueeze(-1) * Km1 + 1.0)).sum(-1)


def calibrate_lambda(FV_train, z_train, K_train):
    """Put the two loss terms on the same scale, by measurement.

    ``lambda`` is set so that a trivial predictor (the training mean of F_V)
    contributes equally to the data term and the physics term.  It is measured
    from the training split only; nothing about it is tuned on validation or
    test data.  Reported in ``results/metrics/train_*.json``.
    """
    with torch.no_grad():
        trivial = FV_train.mean().expand_as(FV_train)
        data_term = ((trivial - FV_train) ** 2).mean().item()
        phys_term = (rachford_rice_torch(trivial, z_train, K_train) ** 2).mean().item()
    if phys_term <= 0.0:
        return 0.0, data_term, phys_term
    return data_term / phys_term, data_term, phys_term


def train_one_epoch(model, optimizer, criterion, data_loader, lam=0.0):
    """Day01 solutions cell 35, with the optional physics term added."""
    model.train()
    tot, tot_data, tot_phys = 0.0, 0.0, 0.0
    for X, y, z, K in data_loader:
        X, y, z, K = X.to(device), y.to(device), z.to(device), K.to(device)
        optimizer.zero_grad()
        pred = model(X)
        data_loss = criterion(pred, y)
        if lam > 0.0:
            phys_loss = (rachford_rice_torch(pred, z, K) ** 2).mean()
        else:
            phys_loss = torch.zeros((), device=X.device)
        loss = data_loss + lam * phys_loss
        loss.backward()
        optimizer.step()
        n = X.size(0)
        tot += loss.item() * n
        tot_data += data_loss.item() * n
        tot_phys += phys_loss.item() * n
    m = len(data_loader.dataset)
    return tot / m, tot_data / m, tot_phys / m


def validate(model, criterion, data_loader, lam=0.0):
    """Day01 solutions cell 35 (``validate``), same accounting."""
    model.eval()
    tot, tot_data, tot_phys = 0.0, 0.0, 0.0
    for X, y, z, K in data_loader:
        with torch.no_grad():
            X, y, z, K = X.to(device), y.to(device), z.to(device), K.to(device)
            pred = model(X)
            data_loss = criterion(pred, y)
            phys_loss = (rachford_rice_torch(pred, z, K) ** 2).mean()
            n = X.size(0)
            tot += (data_loss + lam * phys_loss).item() * n
            tot_data += data_loss.item() * n
            tot_phys += phys_loss.item() * n
    m = len(data_loader.dataset)
    return tot / m, tot_data / m, tot_phys / m


def predict(model, X):
    """Day01 solutions cell 37 (``evaluate``), simplified to one batch."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
