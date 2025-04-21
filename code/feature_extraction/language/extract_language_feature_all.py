import os
import sys
import argparse
import pandas as pd
import subprocess
import torch
import time
from collections import defaultdict
import yaml
import random

def get_gpu_count():
    return torch.cuda.device_count()

def parse_args():
    parser = argparse.ArgumentParser(description="Master script for feature extraction")
    parser.add_argument("--root_dir", type=str, default="../../../data/processed", help="Root data directory")
    parser.add_argument("--subj", type=str, default="subj01", help="Subject identifier")
    parser.add_argument("--model_yaml", type=str, default="models.yaml", help="YAML file with model dict")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--max_tasks", type=int, default=1, help="Task per GPU. ")
    parser.add_argument("--model_start_index", default=0)
    parser.add_argument("--model_end_index", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = f"{args.root_dir}/image_features/{args.subj}"
    os.makedirs(f"{output_dir}/language", exist_ok=True)

    gpu_count = get_gpu_count()
    # gpu_count = 1
    print(f"Detected {gpu_count} GPUs.")

    if gpu_count == 0:
        print("No GPUs detected. Exiting.")
        sys.exit(1)

    max_tasks_per_gpu = args.max_tasks
    gpu_task_limit = {gpu: max_tasks_per_gpu for gpu in range(gpu_count)}

    model_list = yaml.load(open(args.model_yaml, 'r'), Loader=yaml.FullLoader)
    if args.model_end_index is not None:
        model_list = model_list[args.model_start_index:args.model_end_index]

    random.shuffle(model_list)

    def get_available_gpus(required_devices):
        available_gpus = []
        for gpu, available_tasks in gpu_task_limit.items():
            if available_tasks > 0:
                available_gpus.append(gpu)
                if len(available_gpus) == required_devices:
                    return available_gpus
        return None

    processes = []
    model_flag = {list(model_cfg.keys())[0].replace("/", "_"): 0 for model_cfg in model_list}

    while True:
        for model_cfg in model_list:
            model_name = list(model_cfg.keys())[0]
            value = model_cfg[model_name]
            safe_model_name = model_name.replace("/", "_")
            required_devices = value.get("device_count", 1)

            if model_flag[safe_model_name]:
                continue

            if os.path.exists(f"{output_dir}/language/{safe_model_name}/{safe_model_name}.pt"):
                print(f"[SKIP] Features for {safe_model_name} already exist.")
                model_flag[safe_model_name] = True
                continue

            available_gpus = get_available_gpus(required_devices)
            if available_gpus is not None:

                for gpu in available_gpus:
                    gpu_task_limit[gpu] -= 1
                model_flag[safe_model_name] = True

                visible_devices = ",".join(map(str, available_gpus))

                cmd = [
                    sys.executable,
                    "extract_language_feature_single.py",
                    "--model_name", model_name,
                    "--model_path", value["model_path"],
                    "--device_id", "auto",
                    "--output_dir", output_dir,
                    "--subj", args.subj,
                    "--root_dir", args.root_dir,
                    "--batch_size", str(args.batch_size)
                ]

                if 'pythia' in model_name:
                    cmd.extend(["--revision", value['revision']])

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = visible_devices
                process = subprocess.Popen(cmd, env=env)

                processes.append((process, available_gpus, model_name, required_devices))
                print(f"[START] Started processing {model_name} on GPUs {visible_devices}.")
                print(' '.join(cmd))
                print(f"[INFO] {sum(model_flag.values())}/{len(model_flag)} models completed.")


            for proc, gpus, model, devices in processes.copy():
                ret = proc.poll()
                if ret is not None:
                    processes.remove((proc, gpus, model, devices))
                    for gpu in gpus:
                        gpu_task_limit[gpu] += 1
                    if ret == 0:
                        print(f"[DONE] {model} completed successfully on GPUs {','.join(map(str, gpus))}.")
                    else:
                        print(f"[ERROR] {model} failed on GPUs {','.join(map(str, gpus))} with return code {ret}.")

        if all(model_flag.values()):
            break

    print(f"[{args.subj}] All tasks completed.")

if __name__ == '__main__':
    main()