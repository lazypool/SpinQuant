from forward_only.rotation_model import RotateModule
import torch
from torch.optim.optimizer import Optimizer

class PolicyGradientOptimizer(Optimizer):
    def __init__(self, params, modules, lr=1e-3, T=5) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if T <= 0:
            raise ValueError(f"Invalid T value: {T}")
        defaults = dict(lr=lr, T=T)
        super().__init__(params, defaults)
        self.baseline = 0.0
        self.rotate_modules = modules

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        loss = loss - self.baseline

        lr = self.param_groups[0]['lr']
        T = self.param_groups[0]['T']

        for module in self.rotate_modules:
            assert isinstance(module, RotateModule)
            mu_grad, rho_grad = module.compute_grad()
            module.update_param(
                mu = module.mu - lr * loss * mu_grad,
                rho = module.rho - lr * loss * rho_grad
            )

        # update baseline
        self.baseline = (T - 1) / T * self.baseline + (1 / T) * loss

        return loss
