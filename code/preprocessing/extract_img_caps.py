import argparse, os
import json
import PIL
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from nsda import NSDAccess
from PIL import Image
from pycocotools.coco import COCO

def process_image(s):
    try:

        img = nsda.read_images(s)
        img_path = f"{output_img_dir}/nsd_image_{s:06}.png"
        PIL.Image.fromarray(img).save(img_path)
        return s

    except Exception as e:
        print(f"Error processing image {s:06}: {e}")
        return s


if __name__ == "__main__":

    root_dir = '../../data'
    output_img_dir = f'{root_dir}/processed/images'
    os.makedirs(output_img_dir, exist_ok=True)
    output_caption_file = f'{root_dir}/processed/nsd_coco_captions.json'
    output_sup_cls_file = f'{root_dir}/processed/nsd_coco_sup_cls.json'
    output_cls_file = f'{root_dir}/processed/nsd_coco_cls.json'
    num_images = 73000

    coco = COCO(f'{root_dir}/nsddata_stimuli/stimuli/nsd/annotations/instances_val2017.json')
    cats = coco.loadCats(coco.getCatIds())
    cats = {cat['id']: cat for cat in cats}
    nsda = NSDAccess(root_dir)

    instances = nsda.read_image_coco_info(list(range(0, 73000)), info_type='instances')
    instances = [sorted(inst, key=lambda x: x['area'], reverse=True) for inst in instances]

    # get categories
    sup_cls, cls = [], []
    for instance in instances:
        sup_cls.append(cats[instance[0]['category_id']]['supercategory'])
        cls.append(cats[instance[0]['category_id']]['name'])
        # print(sup_cls[-1], cls[-1])

    with open(output_sup_cls_file, 'w') as f:
        json.dump(sup_cls, f, indent=4)

    with open(output_cls_file, 'w') as f:
        json.dump(cls, f, indent=4)

    # # get captions
    prompts = nsda.read_image_coco_info(list(range(0, 73000)), info_type='captions')
    captions = []
    for prompt in prompts:
        captions.append([p['caption'] for p in prompt])

    with open(output_caption_file, 'w') as f:
        json.dump(captions, f, indent=4)


    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_image, s) for s in range(num_images)]
        for future in tqdm(futures, total=num_images):
            s = future.result()

