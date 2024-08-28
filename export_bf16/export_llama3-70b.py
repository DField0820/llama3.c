#!/usr/bin/env python3

import os
import re
import struct
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def extract_layer_info(tensor_name):
    match = re.match(r'layers\.(\d+)\.(.+)', tensor_name)
    if match:
        layer_num = int(match.group(1))
        layer_suffix = match.group(2)
        return layer_num, layer_suffix
    else:
        return None, tensor_name

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="the output dirpath")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pth", type=str, help="pth model path")
    args = parser.parse_args()

    dirpath = args.dirpath
    os.makedirs(dirpath, exist_ok=True)

    # set ModelArgs
    dim = 8192
    hidden_dim = 28672
    n_layers = 80
    n_heads = 64
    n_kv_heads = 8
    vocab_size = -128256
    max_seq_len = 8192
    
    config = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)

    config_filepath = os.path.join(dirpath, f'config.bin')
    with open(config_filepath, 'wb') as f:
        f.write(config)
    print(f"Saved config to {config_filepath}")

    model_path = args.pth
    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, weights_only=True) for p in model_paths]

    for name in list(models[0]):
        

        is_concat = False
        tensors = [model[name] for model in models]

        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            concat_tensor = tensors[0]
            is_concat = True
        is_axis_1 = (
            name.endswith('.attention.wo.weight')
            or name.endswith('.feed_forward.w2.weight')
        )
        axis = 1 if is_axis_1 else 0
        if not is_concat:
            concat_tensor = nn.Parameter(torch.cat(tensors, dim=axis))

        layer_num, layer_suffix = extract_layer_info(name)

        if layer_num is not None:
            layer_path = os.path.join(dirpath, f"layer.{layer_num:03d}")
            os.makedirs(layer_path, exist_ok=True)

            file_name = f"weight.{layer_suffix.replace('.weight', '')}.bin"
            file_path = os.path.join(layer_path, file_name)
            with open(file_path, 'wb') as f:
                serialize_fp32(f, concat_tensor)
                print(f"Saved weights to {file_path}")
        else:
            file_name = f"weight.{layer_suffix.replace('.weight', '')}.bin"
            file_path = os.path.join(dirpath, file_name)
            with open(file_path, 'wb') as f:
                serialize_fp32(f, concat_tensor)
                print(f"Saved weights to {file_path}")

