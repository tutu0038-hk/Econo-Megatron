# Copyright (c) 2023, ALIBABA CORPORATION.  All rights reserved.

"""Pretrain LLaMA"""
import os

from functools import partial

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.LLaMA.llama_model import LLaMAModel
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec, bert_layer_local_spec
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core import mpu, tensor_parallel

import EconoLLM.ReplaceTensor 


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building BERT model ...')
    args = get_args()
    args.transformer_impl = "local"
    args.batch_p2p_comm = True
    config = core_transformer_config_from_args(args)
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = LLaMAModel(
            config=config,
            num_tokentypes=num_tokentypes,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)

    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    #tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    dim = [args.global_batch_size, args.seq_length, args.hidden_size]
    tokens_ = EconoLLM.ReplaceTensor.FetchFakeTensor(dim, datatype.itemsize)

    # Unpack.
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    tokens = EconoLLM.ReplaceTensor.FetchFakeTensor(dim, datatype.itemsize)
    #fixed data size for testing

    return tokens, labels, None, None, None

def loss_func(loss_mask, output_tensor):
    loss = output_tensor.float()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for LLaMA ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating LLaMA datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    # os.environ['NCCL_NET_GDR_READ'] = '0'
    # os.environ['NCCL_NET_GDR_LEVEL'] = '0'
    os.environ['NCCL_MIN_NCHANNELS'] = '16'

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )

    rank = torch.distributed.get_rank()
    EconoLLM.ReplaceTensor.print_trace(rank)
    if rank == 0:
        EconoLLM.ReplaceTensor.solve()
