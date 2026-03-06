# coding=utf-8
# Copyright (c) xiaoxiao lazypool@proton.me
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on SpinQuant(https://github.com/faceresearch/SpinQuant/tree/main/SpinQuant).
# Licensed under Apache License 2.0.

from transformers import Trainer
from forward_only.policy_gradient_optimizer import PolicyGradientOptimizer


class ForwardOnlyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_idx = 0
        self._total_batches = None

    def training_step(self, model, inputs):
        if self._total_batches is None:
            self._total_batches = len(self.get_train_dataloader())

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        loss = self.optimizer.step(lambda: loss)

        return loss.detach()
