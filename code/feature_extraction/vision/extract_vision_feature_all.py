import os
import sys
import argparse
import pandas as pd
import subprocess
import torch
import time
from collections import defaultdict

def get_gpu_count():
    return torch.cuda.device_count()

def parse_args():
    parser = argparse.ArgumentParser(description="Master script for feature extraction")
    parser.add_argument("--root_dir", type=str, default="../../../data/processed", help="Root data directory")
    parser.add_argument("--subj", type=str, default="subj01", help="Subject identifier")
    parser.add_argument("--model_csv", type=str, default="data/results-imagenet.csv", help="CSV file with model list")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--max_tasks", type=int, default=3, help="Task per GPU. ")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--reversed", action="store_true", help="Reverse model list")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = f"{args.root_dir}/image_features/{args.subj}"
    os.makedirs(f"{output_dir}/vision", exist_ok=True)

    gpu_count = get_gpu_count()
    print(f"Detected {gpu_count} GPUs.")

    if gpu_count == 0:
        print("No GPUs detected. Exiting.")
        sys.exit(1)

    # Each GPU can handle up to 3 tasks
    max_tasks_per_gpu = args.max_tasks
    gpu_task_limit = {gpu: max_tasks_per_gpu for gpu in range(gpu_count)}

    if args.model_csv.endswith('.csv'):
        model_csv = pd.read_csv(args.model_csv)
        model_list = list(model_csv['model'])
    else:
        model_names = ['mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl']
        model_list = [f"{model_name}---{x}" for x in [0, 1, 2, 4, 8, 32, 64, 128, 256, 384, 450] for model_name in model_names]
    if args.reversed:
        model_list = model_list[::-1]

    # Function to find the next available GPU
    def get_available_gpu():
        count_max, count_max_gpu = 0, 0
        for gpu, count in gpu_task_limit.items():
            if count > count_max:
                count_max = count
                count_max_gpu = gpu
        if count_max > 0:
            return count_max_gpu
        return None

    processes = []
    model_flag = {model: False for model in model_list}
    while True:
        for model_str in model_list:
            if model_flag[model_str]:
                continue
            if os.path.isfile(f"{output_dir}/vision/{model_str}/{model_str}.pt"):
                print(f"[INFO] Features for {model_str} already exist.")
                model_flag[model_str] = True
                continue

            available_gpu = get_available_gpu()
            if available_gpu is not None:

                # Increment the task count for the selected GPU
                gpu_task_limit[available_gpu] -= 1
                model_flag[model_str] = True

                # Prepare the command to run worker.py
                cmd = [
                    sys.executable,  # Path to the Python interpreter
                    "extract_vision_feature_single.py",
                    "--model_str", model_str,
                    "--device_id", str(available_gpu),
                    "--output_dir", output_dir,
                    "--subj", args.subj,
                    "--root_dir", args.root_dir,
                    "--batch_size", str(args.batch_size)
                ]

                # Start the subprocess
                process = subprocess.Popen(cmd)
                processes.append((process, available_gpu, model_str))
                print(f"[START] Started processing {model_str} on GPU {available_gpu}.")
                print(' '.join(cmd))

            for proc, gpu, model in processes.copy():
                ret = proc.poll()
                if ret is not None:
                    # Process finished
                    processes.remove((proc, gpu, model))
                    gpu_task_limit[gpu] += 1
                    if ret == 0:
                        print(f"[DONE] {model} completed successfully on GPU {gpu}.")
                    else:
                        print(f"[ERROR] {model} failed on GPU {gpu} with return code {ret}.")

        if all(model_flag.values()):
            break

    print("All tasks completed.")

if __name__ == '__main__':
    main()