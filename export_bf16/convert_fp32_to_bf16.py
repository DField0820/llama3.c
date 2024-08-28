#!/usr/bin/env python3                                                                                                                                                                      
import os
import numpy as np
import argparse
import shutil

def convert_fp32_to_bf16(src_file, dest_file):
    print(f"Converting {src_file} to {dest_file}...")

    with open(src_file, 'rb') as f:
        data_fp32 = np.frombuffer(f.read(), dtype='<f4')

    data_uint32 = data_fp32.view('<u4')
    data_bf16 = (data_uint32 >> 16).astype('<u2')

    with open(dest_file, 'wb') as f:
        f.write(data_bf16.tobytes())

def process_directory(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_file_path, src_dir)
            dest_file_path = os.path.join(dest_dir, relative_path)

            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

            if file == 'config.bin':
                print(f"Copying {src_file_path} to {dest_file_path}...")
                shutil.copy2(src_file_path, dest_file_path)
            else:
                convert_fp32_to_bf16(src_file_path, dest_file_path)

def main():
    parser = argparse.ArgumentParser(description="Convert FP32 files to BF16 by truncating lower 16 bits.")
    parser.add_argument('src_dir', type=str, help="Source directory containing FP32 files.")
    parser.add_argument('dest_dir', type=str, help="Destination directory to save BF16 files.")
    args = parser.parse_args()

    process_directory(args.src_dir, args.dest_dir)

if __name__ == "__main__":
    main()
