#!/usr/bin/env python3

import os
import re
import struct
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def convert_tensor(w, n_heads, dim):
    dim1 = len(tensor)
    dim2 = len(tensor[0])
    return (
        w.view(int(n_heads * (dim1 / dim)), 2, int(dim / n_heads / 2), dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )
    
def extract_layer_info(tensor_name):
    match = re.match(r'model.layers\.(\d+)\.(.+)', tensor_name)
    if match:
        layer_num = int(match.group(1))
        layer_suffix = match.group(2)
        return layer_num, layer_suffix
    else:
        return None, tensor_name

def replace_key_names(key):
    replacements = {
        "model.embed_tokens.weight": "weight.tok_embeddings",
        "self_attn.q_proj.weight": "weight.attention.wq",
        "self_attn.k_proj.weight": "weight.attention.wk",
        "self_attn.v_proj.weight": "weight.attention.wv",
        "self_attn.o_proj.weight": "weight.attention.wo",
        "mlp.gate_proj.weight": "weight.feed_forward.w1",
        "mlp.up_proj.weight": "weight.feed_forward.w3",
        "mlp.down_proj.weight": "weight.feed_forward.w2",
        "input_layernorm.weight": "weight.attention_norm",
        "post_attention_layernorm.weight": "weight.ffn_norm",
        "model.norm.weight": "weight.norm",
        "lm_head.weight": "weight.output"
    }

    for old_key, new_key in replacements.items():
        key = key.replace(old_key, new_key)
    return key

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="the output dirpath")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf", type=str, help="hf model path")
    args = parser.parse_args()

    dirpath = args.dirpath
    os.makedirs(dirpath, exist_ok=True)

    # set ModelArgs
    dim = 4096
    hidden_dim = 14336
    n_layers = 32
    n_heads = 32
    n_kv_heads = 8
    vocab_size = -128256
    max_seq_len = 8192
    #max_seq_len = 131072

    config = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)

    config_filepath = os.path.join(dirpath, f'config.bin')
    with open(config_filepath, 'wb') as f:
        f.write(config)
    print(f"Saved config to {config_filepath}")

    # load HF model
    model_path = args.hf
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    hf_dict = hf_model.state_dict()

    for name in list(hf_dict.keys()):
        tensor = hf_dict[name]
        layer_num, layer_suffix = extract_layer_info(name)

        if layer_suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight"):
            tensor = convert_tensor(tensor, n_heads, dim)

        if layer_num is not None:
            layer_path = os.path.join(dirpath, f"layer.{layer_num:03d}")
            os.makedirs(layer_path, exist_ok=True)

            file_name = f"{replace_key_names(layer_suffix)}.bin"
            file_path = os.path.join(layer_path, file_name)
            with open(file_path, 'wb') as f:
                serialize_fp32(f, tensor)
                print(f"Saved weights to {file_path}")
        else:
            file_name = f"{replace_key_names(layer_suffix)}.bin"
            file_path = os.path.join(dirpath, file_name)
            with open(file_path, 'wb') as f:
                serialize_fp32(f, tensor)
                print(f"Saved weights to {file_path}")
