import os
import json

import numpy as np
import torch

if __name__ == '__main__':
    root_dir = '../../data/processed'
    image_ids = {}

    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:
        image_ids[subj] = json.load(open(f"{root_dir}/fmris/{subj}/nsd_fmri2image.json"))['train']

    caption_dict = json.load(open(f"{root_dir}/nsd_coco_captions.json"))

    areas = ['V1']  # 180

    fmris = {}
    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:  # , 'subj07'
        fmris[subj] = {}
        for area in areas:
            data = np.load(f"{root_dir}/fmris/{subj}/area/nsd_{area}_betas_tr.npy")

            fmris[subj][area] = {}
            for fmri_id, image_id in enumerate(image_ids[subj]):
                if fmris[subj][area].get(image_id) is None:
                    fmris[subj][area][image_id] = []
                fmris[subj][area][image_id].append(data[fmri_id])

    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:  # , 'subj07'
        for area in areas:
            for key in fmris[subj][area]:
                fmris[subj][area][key] = np.stack(fmris[subj][area][key]).mean(axis=0)

    # output image configs
    output_dir = f'{root_dir}/index'
    os.makedirs(output_dir, exist_ok=True)

    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:
        outputs = {
            "name": "fmri_scores",
            "description": "To measure the similarity between different neural networks and fMRIs.",
            "data": [],
        }

        image_ids_output = [x for x in fmris[subj]['V1'].keys()]
        for ids in image_ids_output:
            outputs["data"].append({
                "image_id": ids,
                "caption": caption_dict[ids],
            })

        with open(f"{output_dir}/{subj}.json", "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

        # print(len(outputs["data"]))


    output_dir = f'{root_dir}/fmri_pt'
    os.makedirs(output_dir, exist_ok=True)

    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:
        image_ids_output = [x for x in fmris[subj]['V1'].keys()]
        outputs = {}
        for area in areas:
            outputs[area] = torch.stack([torch.tensor(fmris[subj][area][image_id]) for image_id in image_ids_output])
            # print(outputs[area].shape)
        torch.save(outputs, f"{output_dir}/{subj}.pt")
