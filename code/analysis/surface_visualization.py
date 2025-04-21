import json
import os
import yaml
import random
import argparse

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import cortex
import nibabel as nib

plt.rcParams['font.family'] = 'Arial'  #
sns.reset_defaults()
sns.set_theme(context='paper', style='ticks', font='Arial')

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'figure.dpi': 150,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5
})

ROOT_DIR = '/mnt/home/floyed/code_for_brain_align'
DATA_DIR = f"{ROOT_DIR}/data"
SCRIPTS_DIR = f"{ROOT_DIR}/scripts"
FIGURES_DIR = f"{ROOT_DIR}/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def map_scores_to_atlas_vectorized(atlas, region_name_to_score_dict, region_name_to_id):
    """Vectorized version of the mapping function"""
    # Create a mapping array: index is region ID, value is score
    values = list(region_name_to_score_dict.values())
    _mean = np.mean(values)
    _std = np.std(values)
    _min = 1e3
    _max = -1e3

    max_id = max(region_name_to_id.values())

    id_to_score = np.zeros(max_id + 1)

    # Fill in scores for regions that exist in both dictionaries
    for region_name, score in region_name_to_score_dict.items():
        if region_name in region_name_to_id:
            region_id = region_name_to_id[region_name]
            id_to_score[region_id] = score

    score_map = id_to_score[atlas]

    return score_map, (_min, _max)


def main():
    colmap = 'cividis' # 'YlOrBr'  # 'Blues'
    parser = argparse.ArgumentParser(description='Generate brain visualization in various formats')
    parser.add_argument('--format', type=str, default='all',
                        choices=['all', 'png', 'pdf', 'nifti', 'html'],
                        help='Output format: png, pdf, nifti, html, or all (default: all)')
    parser.add_argument('--feature', type=str, default='kernel_size')
    parser.add_argument('--logscale', action='store_true')
    parser.add_argument('--vis', action='store_true')


    args = parser.parse_args()

    yeo7_to_values = {
        "Visual": 1,
        "Somatomotor": 2,
        "Limbic": 3,
        "Dorsal Attention": 4,
        "Ventral Attention": 5,
        "Frontoparietal": 6,
        "Default": 7,
    }

    feature = args.feature
    logscale = args.logscale

    formats_to_generate = []
    if args.format == 'all':
        formats_to_generate = ['png', 'pdf', 'nifti', 'html']
    elif args.format == 'nifti':
        formats_to_generate = ['nii.gz']
    else:
        formats_to_generate = [args.format]

    print(f"Generating output in format(s): {formats_to_generate}")

    language_df = pd.read_csv(f"{FIGURES_DIR}/brain_regions/{feature}/language.csv")
    vision_df = pd.read_csv(f"{FIGURES_DIR}/brain_regions/{feature}/vision.csv")

    subject_replace = {
        'S01': 'subj01',
        'S02': 'subj02',
        'S05': 'subj05',
        'S07': 'subj07',
    }

    language_df['tgt_model'] = language_df['tgt_model'].map(subject_replace)
    vision_df['tgt_model'] = vision_df['tgt_model'].map(subject_replace)

    for subj in ['subj01', 'subj02', 'subj05', 'subj07']:

        # Uncomment these codes when running for the first time
        # cortex.freesurfer.import_subj(subj, freesurfer_subject_dir='./freesurfer')  # initialize the subject
        # cortex.freesurfer.import_flat(subj, 'full')
        # cortex.align.automatic(  # initialize the subject
        #     subj,
        #     'full',
        #     f'/{ROOT_DIR}/data/nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/meanbeta.nii.gz'
        # )

        ref_nifti = nib.load(
            f'/{ROOT_DIR}/data/nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/meanbeta.nii.gz')
        affine = ref_nifti.affine

        atlas = json.load(open(f'/{ROOT_DIR}/data/processed/fmris/{subj}/atlas.json', 'r'))
        atlas, labels = np.array(atlas[0]), atlas[1]
        atlas = np.transpose(atlas, (2, 1, 0)).astype(int)

        for metric in tqdm(language_df['metric'].unique(), desc=subj):
            for score_name in ['scores']:

                language_output_dir = f'{FIGURES_DIR}/brain_regions/{feature}/{subj}/language/{score_name}'
                vision_output_dir = f'{FIGURES_DIR}/brain_regions/{feature}/{subj}/vision/{score_name}'
                delta_output_dir = f'{FIGURES_DIR}/brain_regions/{feature}/{subj}/delta/{score_name}'

                # 添加所需的输出目录
                output_dir_map = {'png': 'png', 'pdf': 'pdf', 'nifti': 'nii.gz', 'html': 'html'}
                for format_type in formats_to_generate:
                    sub_dir = output_dir_map.get(format_type, format_type)
                    os.makedirs(f"{language_output_dir}/{sub_dir}", exist_ok=True)
                    os.makedirs(f"{vision_output_dir}/{sub_dir}", exist_ok=True)
                    os.makedirs(f"{delta_output_dir}/{sub_dir}", exist_ok=True)

                sub_language_df = language_df[(language_df['tgt_model'] == subj) & (language_df['metric'] == metric)]
                sub_vision_df = vision_df[(vision_df['tgt_model'] == subj) & (vision_df['metric'] == metric)]

                sub_language_dict = {
                    row['tgt_feature']: row['score'] for _, row in sub_language_df.iterrows()
                }
                sub_vision_dict = {
                    row['tgt_feature']: row['score'] for _, row in sub_vision_df.iterrows()
                }

                sub_language_score, (_min, _max) = map_scores_to_atlas_vectorized(atlas, sub_language_dict, labels)
                sub_vision_score, (_min, _max) = map_scores_to_atlas_vectorized(atlas, sub_vision_dict, labels)

                epsilon = 1e-3
                language_data = np.maximum(sub_language_score, epsilon)
                vision_data = np.maximum(sub_vision_score, epsilon)

                if logscale:
                    language_data = language_data / np.max(language_data)
                    vision_data = vision_data / np.max(vision_data)

                    language_data = np.log(language_data + epsilon)
                    vision_data = np.log(vision_data + epsilon)

                    suffix = '_log'
                else:
                    suffix = ''

                language_vol = cortex.Volume(
                    language_data,
                    subj,
                    'full',
                    cmap=colmap,
                    logscale=True
                )
                vision_vol = cortex.Volume(
                    vision_data,
                    subj,
                    'full',
                    cmap=colmap,
                    logscale=True
                )

                delta_data = -(language_data - language_data.mean()) / language_data.std() + (vision_data - vision_data.mean()) / vision_data.std()
                delta_vol = cortex.Volume(
                    delta_data,
                    subj,
                    'full',
                    cmap='RdYlBu',
                    vmin=-np.percentile(np.abs(delta_data), 99),
                    vmax=np.percentile(np.abs(delta_data), 99),
                    logscale=True
                )

                if 'png' in formats_to_generate or 'pdf' in formats_to_generate:
                    cortex.quickflat.make_figure(
                        language_vol,
                        sampler='trilinear',
                        with_curvature=True,
                        curvature_threshold=0.8,
                        with_colorbar=False,
                        with_sulci=True,
                        dpi=600,
                        colorbar_location='right'
                    )

                    if 'pdf' in formats_to_generate:
                        plt.savefig(f"{language_output_dir}/pdf/{metric}{suffix}.pdf")
                        print(f"Saved PDF for {subj} - Language - {metric} (log scale)")

                    if 'png' in formats_to_generate:
                        plt.savefig(f"{language_output_dir}/png/{metric}{suffix}.png")
                        print(f"Saved PNG for {subj} - Language - {metric} (log scale)")

                    plt.close()

                    cortex.quickflat.make_figure(
                        vision_vol,
                        sampler='trilinear',
                        with_curvature=True,
                        with_sulci=True,
                        with_colorbar=False,
                        curvature_threshold=0.8,
                        dpi=600,
                        colorbar_location='right'
                    )

                    if 'pdf' in formats_to_generate:
                        plt.savefig(f"{vision_output_dir}/pdf/{metric}{suffix}.pdf")
                        print(f"Saved PDF for {subj} - Vision - {metric} (log scale)")

                    if 'png' in formats_to_generate:
                        plt.savefig(f"{vision_output_dir}/png/{metric}{suffix}.png")
                        print(f"Saved PNG for {subj} - Vision - {metric} (log scale)")

                    cortex.quickflat.make_figure(
                        delta_vol,
                        sampler='trilinear',
                        with_curvature=True,
                        with_sulci=True,
                        with_colorbar=False,
                        curvature_threshold=0.8,
                        dpi=600,
                        colorbar_location='right'
                    )

                    if 'pdf' in formats_to_generate:
                        plt.savefig(f"{delta_output_dir}/pdf/{metric}{suffix}.pdf")
                        print(f"Saved PDF for {subj} - Delta - {metric} (log scale)")

                    if 'png' in formats_to_generate:
                        plt.savefig(f"{delta_output_dir}/png/{metric}{suffix}.png")
                        print(f"Saved PNG for {subj} - Delta - {metric} (log scale)")

                    plt.close()

                if 'nii.gz' in formats_to_generate or 'nifti' in formats_to_generate:
                    language_nifti = nib.Nifti1Image(sub_language_score, affine)
                    nib.save(language_nifti, f"{language_output_dir}/nii.gz/{metric}.nii.gz")
                    print(f"Saved NIfTI for {subj} - Language - {metric}")

                    vision_nifti = nib.Nifti1Image(sub_vision_score, affine)
                    nib.save(vision_nifti, f"{vision_output_dir}/nii.gz/{metric}.nii.gz")
                    print(f"Saved NIfTI for {subj} - Vision - {metric}")

                if 'html' in formats_to_generate:
                    language_html_path = f"{language_output_dir}/html/{metric}{suffix}.html"
                    cortex.webgl.make_static(
                        language_html_path,
                        language_vol,
                        title=f"{subj} - Language - {metric} (Log Scale)",
                        colormap=colmap,
                        colorbar=True,
                        recache=True
                    )

                    vision_html_path = f"{vision_output_dir}/html/{metric}{suffix}.html"
                    cortex.webgl.make_static(
                        vision_html_path,
                        vision_vol,
                        title=f"{subj} - Vision - {metric} (Log Scale)",
                        colormap=colmap,
                        colorbar=True,
                        recache=True
                    )

                    print(f"Generated interactive HTML for {subj} - {metric} (log scale):")
                    print(f" - Language: {language_html_path}")
                    print(f" - Vision: {vision_html_path}")


if __name__ == '__main__':
    main()