import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
from functools import partial
from typing import Dict, Any, Union, Callable, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import PriorityQueue
import multiprocessing
import torch
from tqdm import tqdm

from metrics import AlignmentMetrics


multiprocessing.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic GPU Task Assignment for Feature Alignment.")
    # Data config
    parser.add_argument("--root_dir", type=str, default="../../data/processed", help="Root data directory")
    parser.add_argument("--subj", type=str, default="subj01", help="Subject identifier")
    parser.add_argument("--src_subj", type=str, default=None, help="Subject identifier")
    parser.add_argument("--src", type=str, default="vision", help="Source modality")
    parser.add_argument("--tgt", type=str, default="fmri", help="Target modality")
    parser.add_argument("--metric", type=str, default="cka", help="Metric for feature alignment comparison")
    parser.add_argument("--max-dims", type=int, default=4096, help="Maximum dimensions for SVD reduction")
    # Concurrency config
    parser.add_argument("--max-workers", type=int, default=16, help="Maximum number of concurrent workers")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for evaluation")
    parser.add_argument("--tasks-per-gpu", type=int, default=2, help="Maximum tasks per GPU at any time")
    return parser.parse_args()

def get_feature_filepaths(feature_type, args, subj):

    if feature_type == 'fmri':
        feature_dir = os.path.join(args.root_dir, "fmri_pt")
        return [os.path.join(feature_dir, f"{subj}.pt")]

    feature_dir = os.path.join(args.root_dir, "image_features", subj, feature_type)
    if not os.path.isdir(feature_dir):
        print(f"[WARNING] Directory not found: {feature_dir}")
        return []

    filepaths = []
    for subdir in os.listdir(feature_dir):
        model_path = os.path.join(feature_dir, subdir, f"{subdir}.pt")
        if '---' in model_path:
            continue
            epoch = model_path.split('---')[-1].replace('.pt', '')
            epoch = int(epoch)
            if epoch not in [0, 1, 2, 4, 8, 32, 64, 128, 256, 384, 450]:
                print(f"[WARNING] Skipping {model_path}")
                continue

        elif '--' in model_path:
            continue

        if 'pythia' in model_path:
            continue
            step = model_path.split('_step')[-1].replace('.pt', '')
            step = int(step)
            if step not in [0, 1, 2, 4, 8, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
                print(f"[WARNING] Skipping {model_path}")
                continue

        if os.path.isfile(model_path):
            filepaths.append(model_path)
        else:
            print(f"[WARNING] Skipping {model_path}")
    return filepaths


import torch


def reduce_feature_dim(features: torch.Tensor, max_dims: int) -> torch.Tensor:

    return features

    features = features - features.mean(dim=0)
    features_std = features.std(dim=0)
    features_std[features_std == 0] = 1
    features = features / features_std

    cov_matrix = torch.mm(features.T, features) / (features.size(0) - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    principal_components = eigvecs[:, :max_dims]
    reduced_features = torch.mm(features, principal_components)

    return reduced_features


def validate(
    src: Dict[str, Any],
    tgt: Dict[str, Any],
    metric_fn: Callable,
    max_dims: int,
    device: str
) -> Dict[str, Any]:

    results = []
    eps = 1e-8
    with torch.no_grad():
        for src_key, src_value in src.items():
            src_value = reduce_feature_dim(src_value.float().to(device), max_dims)
            src_value = (src_value - src_value.mean(0, keepdim=True)) / (src_value.std(0, keepdim=True) + eps)

            res = []
            for tgt_key, tgt_value in tgt.items():
                tgt_value = reduce_feature_dim(tgt_value.float().to(device), max_dims)
                tgt_value = (tgt_value - tgt_value.mean(0, keepdim=True)) / (tgt_value.std(0, keepdim=True) + eps)

                score = metric_fn(src_value, tgt_value)
                res.append(score)
            results.append(res)
    return {
        'src_features': list(src.keys()),
        'tgt_features': list(tgt.keys()),
        'scores': torch.tensor(results)
    }

def run_validation_task(
        src_filepath: str,
        tgt_filepath: str,
        metric_fn: Callable,
        max_dims: int,
        gpu_enabled: bool,
        gpu_id: Union[int, None],
        output_dir: str
):

    src_filename = os.path.basename(src_filepath).replace(".pt", "")
    tgt_filename = os.path.basename(tgt_filepath).replace(".pt", "")

    output_subdir = os.path.join(output_dir, f"{src_filename}/{tgt_filename}")
    output_filename = os.path.join(output_subdir, "results.pt")


    device = f"cuda:{gpu_id}" if (gpu_enabled and gpu_id is not None) else "cpu"

    try:
        src = torch.load(src_filepath, weights_only=True)
        tgt = torch.load(tgt_filepath, weights_only=True)
        results = validate(src, tgt, metric_fn, max_dims, device)
    except Exception as e:
        print(
            f"[ERROR] Exception during validation. "
            f"src: {src_filename}, "
            f"tgt:{tgt_filename}, "
            f"output: {output_filename}, "
            f"erroe: {e}"
        )
        return gpu_id

    outputs = {
        'src_filename': src_filename,
        'tgt_filename': tgt_filename,
        'results': results
    }

    if torch.isinf(results['scores']).any():
        print(f"[ERROR] Inf values in results: {src_filename} -> {tgt_filename}, Discarding results.")
        return gpu_id

    os.makedirs(output_subdir, exist_ok=True)
    torch.save(outputs, output_filename)
    print(f"[SUCCESS] Results saved to {output_filename}")
    return gpu_id


if __name__ == '__main__':
    args = parse_args()

    src_filepaths = get_feature_filepaths(args.src, args, args.src_subj if args.src_subj else args.subj)
    tgt_filepaths = get_feature_filepaths(args.tgt, args, args.subj)

    print(f"[INFO] Comparing {args.src} -> {args.tgt} for {args.subj}")
    print(f"[INFO] Found {len(src_filepaths)} source features and {len(tgt_filepaths)} target features.")

    if args.src_subj is not None:
        prefix = args.src_subj
    else:
        prefix = args.src
    output_dir = os.path.join(args.root_dir, "comparison", args.subj, f"{prefix}_{args.tgt}/{args.metric}")
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {}
    func_name = args.metric
    if 'nn' in args.metric:
        parts = args.metric.split('_')
        if parts[-1].isdigit():
            kwargs['topk'] = int(parts[-1])
            func_name = '_'.join(parts[:-1])
        else:
            kwargs['topk'] = 12
    if 'cca' in args.metric:
        kwargs['cca_dim'] = 6
    if 'kernel' in args.metric:
        kwargs['dist'] = 'sample'
    if 'cka' in args.metric:
        if '_' in args.metric:
            kwargs['kernel_metric'] = 'rbf'
            kwargs['rbf_sigma'] = float(args.metric.split('_')[-1])
            func_name = 'unbiased_cka' if 'unbiased' in args.metric else 'cka'

    func = getattr(AlignmentMetrics, func_name)
    metric_fn = partial(func, **kwargs)

    gpu_status = {}
    if args.gpu:
        num_gpus = torch.cuda.device_count()
        gpu_status = {i: args.tasks_per_gpu for i in range(num_gpus)}

    tasks = [
        (src_filepath, tgt_filepath)
        for src_filepath in src_filepaths
        for tgt_filepath in tgt_filepaths
    ]

    pending_tasks = tasks.copy()
    ongoing_futures = []

    total_task = len(tasks)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        while pending_tasks or ongoing_futures:
            done_futures = [f for f in ongoing_futures if f.done()]
            for f in done_futures:
                finished_gpu_id = f.result()
                if args.gpu and finished_gpu_id is not None:
                    gpu_status[finished_gpu_id] += 1
                ongoing_futures.remove(f)

            if pending_tasks:
                available = []
                if args.gpu:
                    for gpu_id, cap in gpu_status.items():
                        if cap > 0:
                            prop = torch.cuda.get_device_properties(gpu_id)
                            total_mem = prop.total_memory
                            allocated_mem = torch.cuda.memory_allocated(gpu_id)
                            free_mem = total_mem - allocated_mem
                            if free_mem / total_mem > 0.8:
                                available.append((gpu_id, cap))
                else:
                    available = [(None, None)]

                if available:
                    if args.gpu:
                        selected_gpu_id, _ = min(available, key=lambda x: x[1])
                    else:
                        selected_gpu_id = None

                    if args.gpu and selected_gpu_id is not None:
                        gpu_status[selected_gpu_id] -= 1

                    while len(pending_tasks) > 0:
                        src_filepath, tgt_filepath = pending_tasks.pop(0)
                        src_filename = os.path.basename(src_filepath).replace(".pt", "")
                        tgt_filename = os.path.basename(tgt_filepath).replace(".pt", "")

                        output_subdir = os.path.join(output_dir, f"{src_filename}/{tgt_filename}")
                        output_filename = os.path.join(output_subdir, "results.pt")

                        if os.path.exists(output_filename):
                            print(f"[SKIP] Results already exist: {output_filename}")
                            continue
                        else:
                            break

                    print(f"[Pending / Ongoing / Total] {len(pending_tasks)} / {len(ongoing_futures)} / {total_task}, "
                          f"Processing {src_filepath} -> {tgt_filepath} on GPU {selected_gpu_id}")

                    future = executor.submit(
                        run_validation_task,
                        src_filepath,
                        tgt_filepath,
                        metric_fn,
                        args.max_dims,
                        args.gpu,
                        selected_gpu_id,
                        output_dir
                    )
                    future.gpu_id = selected_gpu_id
                    ongoing_futures.append(future)

            time.sleep(0.1)

    print("[INFO] All tasks have been processed.")