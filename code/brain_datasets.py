# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/6/27 16:43
# User      : floyed
# Product   : PyCharm
# Project   : fmri_score
# File      : brain_datasets.py
# explain   :

import os
import sys
import copy
import json
from typing import Dict, List, Optional, Union, Any

import numpy as np

import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.nn import functional as F
from PIL import Image

from torch.utils.data.dataset import Dataset
import transformers

chat_template = [
    {
        "role": "user",
        "content": "Consider the image described by the following sentences: \n\n'{captions}'"
    },
]

class BrainFeatureDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        subject: str = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        transforms: Any = None,
        max_length: int = 512,
        network_str: Union[str, List[str]] = None,
        requires_all_caps: bool = True,
    ):
        super(BrainFeatureDataset, self).__init__()
        config = json.load(open(f"{data_path}/index/{subject}.json"))

        self.captions = []
        self.images = [f"{data_path}/images/nsd_image_{x['image_id']:06}.png" for x in config["data"]]

        if transforms:
            self.images = [Image.open(x).convert("RGB") for x in self.images]

        if tokenizer:
            self.captions = [x["caption"] for x in config["data"]]

        # print(tokenizer, self.images, self.captions)

        if network_str is None:
            network_str = []
        if isinstance(network_str, str):
            network_str = [network_str]

        self.networks = {}
        for network in network_str:
            fname = f"{data_path}/{network}/{network}.pt"
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Feature file {fname} not found.")
            self.networks[network] = torch.load(fname)

            for feat, value in self.networks[network].items():
                if not value.shape[0] == len(self.images):
                    raise ValueError(f"{network}-{feat} has shape {value.shape}, expected ({len(self.images)}, ...).")

        self.tokenizer = tokenizer
        self.image_transforms = transforms
        self.max_length = max_length
        self.requires_all_caps = requires_all_caps

        if hasattr(tokenizer, 'apply_chat_template'):
            self.chat_template = self.tokenizer.apply_chat_template(
                chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            self.chat_template = '{captions}'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i) -> Dict[str, Any]:
        image, tokens = [], []

        if self.image_transforms is not None:
            # image = Image.open(self.images[i]).convert("RGB")
            # image = image.convert("RGB")
            image = self.images[i]
            image = self.image_transforms(image)

        if self.tokenizer:
            caption = self.captions[i]

            if not self.requires_all_caps:
                caption = caption[0]
            else:
                caption = "\n".join(caption) + "\n"

            caption = self.chat_template.format(captions=caption)
            # print([caption])

            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # print(tokens)
            tokens = {key: value.squeeze(0) for key, value in tokens.items()}

            # features = {}
            # for network, value in self.networks.items():
            #     features[network] = {feat: value[feat][i] for feat in value}

        return {
            "image": image,
            "caption": tokens,
            # "features": features,
        }


if __name__ == '__main__':
    dataset = BrainFeatureDataset(data_path='/mnt/data/datasets/brain_feature', network_str='nsd_subj01')
    print(dataset[0]['features']['nsd_subj01']['early'].shape)
