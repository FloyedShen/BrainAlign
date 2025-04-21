
import json
import os
import sys

sys.path.append('../../')
import random
import argparse

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModel, GPTNeoXForCausalLM
from transformers.pipelines import TextGenerationPipeline
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from brain_datasets import BrainFeatureDataset

PYTHIA_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'user' %}
<user>: {{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
<assistant>: {{ message['content'] }}
{%- endif %}
{%- endfor %}"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Feature Extraction Worker")

    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model name")

    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--device_id", type=str, required=True, help="GPU device ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--subj", type=str, required=True, help="Subject identifier")
    parser.add_argument("--root_dir", type=str, default="../../../data/processed", help="Root data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")

    args = parser.parse_args()

    args.safe_model_name = args.model_name.replace("/", "_")
    output_path = f"{args.output_dir}/language/{args.safe_model_name}"

    if os.path.exists(f"{output_path}/{args.safe_model_name}.pt"):
        print(f"[SKIP] {output_path}/{args.safe_model_name}.pt exists. Skipping...")
        sys.exit()

    if 'pythia' in args.model_name:
        # load tokenizer & dataset
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/" + args.model_name.split("_")[0] if 'pythia' in args.model_name else args.model_name,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True if 'pythia' in args.model_name else False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if 'pythia' in args.model_name:
            tokenizer.chat_template = PYTHIA_TEMPLATE

    else:
        # load tokenizer & dataset
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=False,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.padding_side != "left":
        print(f"[WARNING] Padding side is {tokenizer.padding_side}. This may cause issues, setting to left.")
        tokenizer.padding_side = "left"

    dataset = BrainFeatureDataset(
        data_path=args.root_dir,
        subject=args.subj,
        tokenizer=tokenizer,
        # chat_template=chat_template,
        network_str=None,
        requires_all_caps=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # load model
    retries = 10
    model = None
    while retries > 0:
        try:
            print(args.model_path, args.revision, args.device_id)
            if 'pythia' in args.model_name:
                model = GPTNeoXForCausalLM.from_pretrained(
                    "EleutherAI/" + args.model_name.split("_")[0],
                    cache_dir=args.model_path,
                    revision=args.revision,
                    torch_dtype=torch.float16,
                    device_map=args.device_id,
                    low_cpu_mem_usage=True,
                    # trust_remote_code=True,
                    # local_files_only=True,
                    # access_token=access_token,
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.float16,
                    device_map=args.device_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    # access_token=access_token,
                ) #.eval()
            break
        except Exception as e:
            if retries == 0:
                raise e
            retries -= 1
            print(f"[WARNING] Raising {e} while Loading {args.model_name}, Retrying... {retries} left")

    if model is None:
        raise ValueError(f"Cannot Load model {args.safe_model_name}")
    features = {i: [] for i in range(model.config.num_hidden_layers + 1)}

    desc_str = args.model_name.split("/")[-1]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_str):
            tokens = batch.get('caption', None)
            device = list(model.hf_device_map.values())[0]
            tokens = {key: value.to(device) for key, value in tokens.items()}


            outputs = model(
                **tokens,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )

            for i in range(model.config.num_hidden_layers + 1):
                feature = outputs.hidden_states[i]

                feature = feature[:, -1, :]
                features[i].append(feature.cpu())

    del model
    torch.cuda.empty_cache()

    for key, value in features.items():
        features[key] = torch.cat(value, dim=0)

    os.makedirs(output_path, exist_ok=True)
    torch.save(features, f"{output_path}/{args.safe_model_name}.pt")
    print(f"[INFO] Saved {args.safe_model_name} to {output_path}/{args.safe_model_name}.pt")