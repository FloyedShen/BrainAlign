import os
import sys
import json
import torch
import timm
from tqdm import tqdm
from PIL import Image
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters

sys.path.append('../..')

from brain_datasets import BrainFeatureDataset
import argparse


resume_dict = {
    'mixnet_s': '20250323-022002-mixnet_s-224',
    'mixnet_m': '20250317-220537-mixnet_m-224',
    'mixnet_l': '20250323-152153-mixnet_l-224',
    'mixnet_xl': '20250317-142547-mixnet_xl-224',
}

def exact_feature(model_str, device_id, output_dir, subj, root_dir, batch_size):
    if os.path.isfile(f"{output_dir}/vision/{model_str}/{model_str}.pt"):
        print(f"[INFO] Features for {model_str} already exist.")
        return

    print(f"[INFO] Extracting features for {model_str} on device: {device_id}...")
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    if '---' in model_str:
        model_str, epoch = model_str.split('---')
    else:
        epoch = None
    try:
        model = timm.create_model(model_str, pretrained=not epoch)

        if epoch:
            ckpt = f"/mnt/home/floyed/brain-sim/scripts/vision_model_train/output/train/{resume_dict[model_str]}/checkpoint-{epoch}.pth.tar"
            if not os.path.isfile(ckpt):
                print(f"[WARNING] Checkpoint {ckpt} not found.")
                return
            load_checkpoint(
                model,
                ckpt,
                use_ema=True,
            )
            print(f"[INFO] Loaded model {args.model_str} from epoch {epoch}.")

        model = model.eval().to(device)

        hidden_states = {}

        available_layers_names = ['blocks', 'features', 'stage', 's', 'layer', 'layers', 'block', 'stages']

        def get_intermediate_outputs(name):
            def hook(module, inputs, outputs):
                if isinstance(outputs, list) or isinstance(outputs, tuple):

                    outputs = outputs[0]
                if len(outputs.shape) == 3:
                    hidden_states[name] = outputs.mean(1)
                elif len(outputs.shape) == 4:
                    hidden_states[name] = outputs.mean([-1, -2])
                elif len(outputs.shape) == 2:
                    hidden_states[name] = outputs
                else:
                    while len(outputs.shape) > 2:
                        outputs = outputs.mean(-1)
                    hidden_states[name] = outputs

            return hook

        found = False
        for layer_name in available_layers_names:
            if hasattr(model, layer_name):
                if hasattr(getattr(model, layer_name), '__getitem__'):
                    for idx, layer in enumerate(getattr(model, layer_name)):
                        name = f"{layer_name}_{idx}"
                        layer.register_forward_hook(get_intermediate_outputs(name))
                    found = True
                elif isinstance(getattr(model, layer_name), torch.nn.Module):
                    idx = 1
                    while hasattr(model, f"{layer_name}{idx}"):
                        name = f"{layer_name}{idx}"
                        getattr(model, f"{layer_name}{idx}").register_forward_hook(get_intermediate_outputs(name))
                        idx += 1
                    if idx > 1:
                        found = True

        if not found:
            print(f"[WARNING] Model {args.model_str} has no available layers")
            return

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        print(f"[INFO] Weight Loaded {args.model_str}.")

        dataset = BrainFeatureDataset(
            data_path=root_dir,
            subject=subj,
            tokenizer=None,
            transforms=transforms,
            network_str=None,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        )

        results = {}

        for batch in tqdm(dataloader, desc=f"Extracting features {args.model_str}"):
            images = batch.get('image', None).to(device)
            with torch.no_grad():
                feats = model.forward_features(images)
                res = model.forward_head(feats, pre_logits=True)
            for key, value in hidden_states.items():
                if key not in results:
                    results[key] = []
                results[key].append(value.cpu())

        for key, value in results.items():
            results[key] = torch.cat(value, dim=0)

        output_path = f"{output_dir}/vision/{args.model_str}"
        os.makedirs(output_path, exist_ok=True)
        torch.save(results, f"{output_path}/{args.model_str}.pt")

        print(f"[FINISH] Saved features to {output_path}/{args.model_str}.pt")
        del model, results, dataloader, dataset  # Clear memory
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[Error] {e} {args.model_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Extraction Worker")
    parser.add_argument("--model_str", type=str, required=True, help="Model name")
    parser.add_argument("--device_id", type=int, required=True, help="GPU device ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--subj", type=str, required=True, help="Subject identifier")
    parser.add_argument("--root_dir", type=str, default="../../../data/processed", help="Root data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")

    args = parser.parse_args()

    exact_feature(
        model_str=args.model_str,
        device_id=args.device_id,
        output_dir=args.output_dir,
        subj=args.subj,
        root_dir=args.root_dir,
        batch_size=args.batch_size
    )