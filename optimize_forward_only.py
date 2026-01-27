# coding=utf-8
# Copyright (c) xiaoxiao lazypool@proton.me
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on SpinQuant(https://github.com/faceresearch/SpinQuant/tree/main/SpinQuant).
# Licensed under Apache License 2.0.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from utils.data_utils import CustomJsonDataset
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

from forward_only.rotation_model import RotateModule
from forward_only.policy_gradient_optimizer import PolicyGradientOptimizer
from forward_only.forward_only_trainer import ForwardOnlyTrainer

log: Logger = get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = prepare_model(ptq_args, model)
    model.R1 = RotateModule(model.config.hidden_size, "cuda")
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        model.model.layers[i].self_attn.R2 = RotateModule(
            model.config.hidden_size // model.config.num_attention_heads, "cuda"
        )
    for param in model.parameters():
        param.requires_grad = False
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    trainable_modules = [model.R1] + [
        model.model.layers[i].self_attn.R2
        for i in range(model.config.num_hidden_layers)
    ]
    trainable_parameters = [model.R1.weight] + [
        model.model.layers[i].self_attn.R2.weight
        for i in range(model.config.num_hidden_layers)
    ]
    model.seqlen = training_args.model_max_length
    optimizer = PolicyGradientOptimizer(trainable_parameters, trainable_modules, lr=training_args.learning_rate)
    MyTrainer = ForwardOnlyTrainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    torch.distributed.barrier()

    trainer.train()
    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if "R1.weight" in key or "self_attn.R2" in key
    }
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "R_forward_only.bin")
        torch.save(
            R_dict,
            path,
        )
    dist.barrier()


if __name__ == "__main__":
    train()
