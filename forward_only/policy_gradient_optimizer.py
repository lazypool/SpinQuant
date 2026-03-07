# coding=utf-8
# Copyright (c) xiaoxiao lazypool@proton.me
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on SpinQuant(https://github.com/faceresearch/SpinQuant/tree/main/SpinQuant).
# Licensed under Apache License 2.0.

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

class PolicyGradientOptimizer(Optimizer):
    def __init__(self, params, modules, lr=1e-3, T=5, N=10) -> None:
        defaults = dict(lr=lr)
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
            mu_grad, rho_grad = module.compute_grad()
            grads.append((mu_grad, rho_grad))
        self.samples.append({'loss': loss, 'grads': grads})

        if dist.get_rank() == 0:
            print(f"[Rank 0] Module 0 mu = {self.rotate_modules[0].mu.mean().item()}")
            print(f"[Rank 0] Module 0 rho = {self.rotate_modules[0].rho.mean().item()}")

        if len(self.samples) == N:
            # calculate average loss
            avg_loss = sum(sample['loss'] for sample in self.samples) / N
            if dist.is_available() and dist.is_initialized():
                avg_loss_ = torch.tensor(avg_loss, device='cuda')
                dist.all_reduce(avg_loss_, op=dist.ReduceOp.AVG)
                avg_loss = avg_loss_.item()

            # update baseline
            self.baseline = (T - 1) / T * self.baseline + (1 / T) * avg_loss

            # calculate average grad with baseline
            avg_grads = [[0.0, 0.0] for _ in range(len(self.rotate_modules))]
            for sample in self.samples:
                loss_ = sample['loss'] - self.baseline
                for i, grad in enumerate(sample['grads']):
                    avg_grads[i][0] += loss_ * grad[0] # μ
                    avg_grads[i][1] += loss_ * grad[1] # ρ
            for i in range(0, len(self.rotate_modules)):
                avg_grads[i][0] /= N
                avg_grads[i][1] /= N
            if dist.is_available() and dist.is_initialized():
                for i in range(len(self.rotate_modules)):
                    mu_grad_ = avg_grads[i][0].cuda()
                    rho_grad_ = avg_grads[i][1].cuda()
                    dist.all_reduce(mu_grad_, op=dist.ReduceOp.AVG)
                    dist.all_reduce(rho_grad_, op=dist.ReduceOp.AVG)
                    avg_grads[i][0] = mu_grad_
                    avg_grads[i][1] = rho_grad_

            # update params
            for module, grad in zip(self.rotate_modules, avg_grads):
                module.update_param(
                    module.mu - lr * grad[0],
                    module.rho - lr * grad[1]
                )
            self.samples.clear()

        return loss
