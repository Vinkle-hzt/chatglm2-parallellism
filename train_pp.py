import os.path

import torch
from modeling_chatglm import ChatGLMForConditionalGeneration
from torch.nn import CrossEntropyLoss
import argparse
import math
import json
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import SchedulerType, default_data_collator, get_scheduler
from tokenization_chatglm import ChatGLMTokenizer
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn as nn
import deepspeed
from typing import List, Tuple, Any
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def get_masks(input_ids, past_key_values=None, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    past_length = 0
    if past_key_values:
        past_length = past_key_values[0][0].shape[0]
    if past_length:
        full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                    device=input_ids.device), full_attention_mask), dim=-1)
    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    if not past_length and padding_mask is not None:
        full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

def get_position_ids(input_ids, mask_positions, device):
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(150004) for seq in input_ids]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    for i, context_length in enumerate(context_lengths):
        position_ids[i, context_length:] = mask_positions[i]
    block_position_ids = [torch.cat((torch.zeros(context_length, dtype=torch.long, device=device),
                                     torch.arange(seq_length - context_length, dtype=torch.long,
                                                  device=device) + 1
                                     )) for context_length in context_lengths]
    block_position_ids = torch.stack(block_position_ids, dim=0)
    position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    return position_ids


class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.embedding = model.transformer.embedding

    def forward(self, ipt):
        input_ids, labels = ipt
        inputs_embeds = self.embedding(input_ids)
        # TODO: add back attention mask
        # attention_mask = None
        # _, seq_length = input_ids.shape
        # rotary_pos_emb = self.rotary_pos_emb(seq_length)
        # rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        # rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        # print(f"rotary_pos_emb shape: {rotary_pos_emb.shape}")
        # print(f"inputs_embeds shape: {inputs_embeds.shape}")
        return inputs_embeds, labels


class GLMBlockPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration, layer_idx):
        super().__init__()
        self.rotary_pos_emb = model.transformer.rotary_pos_emb
        self.layer = model.transformer.encoder.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        hidden_states, labels = ipt
        # check if attention_mask is empty
        seq_length = hidden_states.shape[0]
        rotary_pos_emb = self.rotary_pos_emb(seq_length)
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        layer_ret = self.layer(
                    hidden_states,
                    None,
                    rotary_pos_emb,
                    None,
                    False
                )
        hidden_states, _ = layer_ret
        # TODO: add attention mask
        return hidden_states, labels


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.final_layernorm = model.transformer.encoder.final_layernorm

    def forward(self, ipt):
        # TODO: add attention mask
        hidden_states, labels = ipt
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, labels


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.output_layer = model.transformer.output_layer
        self.weight = self.output_layer.weight

    def forward(self, ipt):
        hidden_states, labels = ipt
        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # print(f"lm_logits shape: {lm_logits.shape}")
        # print(f"labels shape: {labels.shape}")
        # print(f"shift_logits shape: {shift_logits.shape}")
        # print(f"shift_labels shape: {shift_labels.shape}")
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
        print(f"[{time.time()}] loss: {loss.item()}")
        return loss


def get_model(model):
    layers = [LayerSpec(EmbeddingPipeLayer, model=model),
              *[LayerSpec(GLMBlockPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.config.num_layers)],
              LayerSpec(FLNPipeLayer, model=model),
              LayerSpec(LMPipeLayer, model=model)]
    return layers

def get_seq_model(model):
    models = [EmbeddingPipeLayer(model), *[GLMBlockPipeLayer(model, idx) for idx in range(model.config.num_layers)],
                FLNPipeLayer(model), LMPipeLayer(model)]
    return nn.Sequential(*models)

def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_path", default="", type=str, help="")
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")

    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=512, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")

    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    parser.add_argument("--save_model_step", default=40, type=int, help="")
    parser.add_argument("--num_stages", default=4, type=int, help="")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
        self.all_data = []
        skip_data_number = 0
        
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                prefix = prompt_text
                prompt = tokenizer.build_prompt(sample["text"])
                answer = sample["answer"]
                prompt = prefix + prompt
                skip_flag = False
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                         max_length=max_src_len)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                         max_length=max_src_len)
                max_tgt_len = max_len - len(a_ids) - 1
                
                if len(a_ids) > max_src_len:
                    a_ids = a_ids[:max_src_len]
                    skip_flag = True
                if len(b_ids) > max_tgt_len:
                    b_ids = b_ids[:max_tgt_len]
                    skip_flag = True
            
                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
                
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue

                self.all_data.append(
                    {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)})
        print("the number of skipping data is {}, the proportion is {}".format(skip_data_number, skip_data_number / (
                len(self.all_data) + skip_data_number)))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance



class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""

    def __call__(self, samples):
        input_ids_list, labels_list = [], []
        for instance in samples:
            input_ids_list.append(instance["input_ids"])
            labels_list.append(instance["labels"])
        return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))


def collect_fn_glm(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["labels"])
    return ((pad_sequence(input_ids_list, batch_first=True), pad_sequence(labels_list, batch_first=True)),
            pad_sequence(labels_list, batch_first=True))


def main():
    args = set_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed(dist_backend="nccl")

    args.global_rank = torch.distributed.get_rank()

    ds_config = {"train_micro_batch_size_per_gpu": args.per_device_train_batch_size / args.gradient_accumulation_steps,
                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "optimizer": {
                     "type": "Adam",
                     "params": {
                         "lr": 2e-5,
                         "betas": [
                             0.9,
                             0.95
                         ],
                         "eps": 1e-8,
                         "weight_decay": 5e-4
                     }
                 },
                 "fp16": {
                     "enabled": True
                 },
                 "zero_optimization": {
                     "stage": 1,
                    #  "offload_optimizer": {
                    #      "device": "cpu",
                    #      "pin_memory": True
                    #  },
                     "allgather_partitions": True,
                     "allgather_bucket_size": 2e8,
                     "overlap_comm": True,
                     "reduce_scatter": True,
                     "reduce_bucket_size": 2e8,
                     "contiguous_gradients": True
                 },
                 "steps_per_print": 5
                 }

    set_random_seed(args.seed)

    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)

    print_rank_0(f"tokenizer special_tokens: {tokenizer.all_special_tokens}", args.global_rank)

    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=128, lora_alpha=64, lora_dropout=0.1
    )
    model_peft = get_peft_model(model, peft_config)
    
    model_pipe = PipelineModule(layers=get_model(model_peft.base_model.model), num_stages=args.num_stages)
    model_pipe.to(device).half()

    train_dataset = GLMPromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    data_collator = DataCollatorForPromptDataset()

    g = torch.Generator()
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=args.per_device_train_batch_size,
                                  generator=g)

    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)
    print_rank_0("args.per_device_train_batch_size = {}".format(args.per_device_train_batch_size), args.global_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)

    train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
    engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=ds_config, model_parameters=model_pipe.parameters())
    start = time.time()
    all_loss = 0.0
    for step in range(args.num_train_epochs * num_update_steps_per_epoch):
        print_rank_0(f"[{time.time()}] start step {step}", args.global_rank)
        loss = engine.train_batch(data_iter=train_dataloader)
        print_rank_0(f"[{time.time()}] step = {step}, loss = {loss.item()}", args.global_rank)
        all_loss += loss.item()
        if args.local_rank == 0:
            if (step + 1) % args.show_loss_step == 0:
                now = time.time()
                avg_time = (now - start) / args.show_loss_step
                avg_loss = all_loss / args.show_loss_step
                print(f"Step={step:>6}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
                start = now
                all_loss = 0.0

        if (step + 1) % args.save_model_step == 0:
            print(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()