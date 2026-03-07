# coding=utf-8
# Copyright (c) xiaoxiao lazypool@proton.me
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on SpinQuant(https://github.com/faceresearch/SpinQuant/tree/main/SpinQuant).
# Licensed under Apache License 2.0.

import torch
from torch import nn
from torch.nn.functional import softplus, sigmoid

class RotateModule(nn.Module):
    def __init__(self, R_init) -> None:
        super(RotateModule, self).__init__()
        n, device = R_init.shape[0], torch.device("cuda")
        self.m, self.n = n * (n - 1) // 2, n
        self.R_init = R_init.to(torch.float32).to(device)

        self.mu = torch.zeros(self.m).to(torch.float32).to(device)
        self.rho = torch.zeros(self.m).to(torch.float32).to(device)
        self.weight = nn.Parameter(self.get_rotation())

    @torch.no_grad()
    def forward(self, x, transpose=False):
        self.weight.data.copy_(self.get_rotation())
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

    def get_rotation(self):
        noise = torch.randn(self.m, device=self.mu.device, dtype=torch.float32)
        a = self.mu + noise * softplus(self.rho) # softplus
        self.a_cache = a.detach() # used to compute grad
        
        # generate skew matrix
        A = torch.zeros(self.n, self.n, device=a.device, dtype=a.dtype)
        rows, cols = torch.triu_indices(self.n, self.n, offset=1)
        A[rows, cols] = a
        A = A - A.T

        I = torch.eye(self.n, device=A.device, dtype=A.dtype)
        R = (I + A) @ torch.inverse(I - A) # cayley-transform

        # return self.R_init @ R
        return R

    def compute_grad(self):
        diff = self.a_cache - self.mu
        sigma = softplus(self.rho) + 1e-3
        me = sigmoid(self.rho) / sigma

        grad_mu = diff / sigma**2
        grad_rho = me * (diff**2 / sigma**2 - 1)

        return grad_mu, grad_rho

    def update_param(self, mu, rho):
        mu = mu.to(self.mu.device)
        rho = rho.to(self.rho.device)
        mu = torch.clamp(mu, -6.0, 6.0)
        rho = torch.clamp(rho, 0.0, 2.0)
        self.mu.data.copy_(mu)
        self.rho.data.copy_(rho)
