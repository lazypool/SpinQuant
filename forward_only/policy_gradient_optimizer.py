# coding=utf-8
# Copyright (c) xiaoxiao lazypool@proton.me
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on SpinQuant(https://github.com/faceresearch/SpinQuant/tree/main/SpinQuant).
# Licensed under Apache License 2.0.

from forward_only.rotation_model import RotateModule
import torch
from torch.optim.optimizer import Optimizer

class PolicyGradientOptimizer(Optimizer):
    def __init__(self, params, modules, lr=1e-3, T=5, N=100) -> None:
        defaults = dict()
        super().__init__(params, defaults)
        self.lr, self.T, self.N = lr, T, N
        self.baseline = 0.0
        self.rotate_modules = modules
        self.samples = list()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        lr, T, N = self.lr, self.T, self.N

        # sampling on all modules
        grads = list()
        for module in self.rotate_modules:
            assert isinstance(module, RotateModule)
            mu_grad, rho_grad = module.compute_grad()
            grads.append((mu_grad, rho_grad))
        self.samples.append({'loss': loss, 'grads': grads})

        if len(self.samples) == N:
            # calculate average loss
            avg_loss = sum(sample['loss'] for sample in self.samples) / N

            # update baseline
            self.baseline = (T - 1) / T * self.baseline + (1 / T) * avg_loss

            # calculate average grad with baseline
            avg_grads = [[0.0, 0.0] for _ in range(0, len(self.rotate_modules))]
            for sample in self.samples:
                loss_ = sample['loss'] - self.baseline
                for i, grad in enumerate(sample['grads']):
                    avg_grads[i][0] += loss_ * grad[0] # μ
                    avg_grads[i][1] += loss_ * grad[1] # ρ
            for i in range(0, len(self.rotate_modules)):
                avg_grads[i][0] /= N
                avg_grads[i][1] /= N

            # update params
            for module, grad in zip(self.rotate_modules, avg_grads):
                assert isinstance(module, RotateModule)
                module.update_param(
                    module.mu - lr * grad[0],
                    module.rho - lr * grad[1]
                )
            self.samples.clear()

        return loss
