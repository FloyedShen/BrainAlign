import json
import os
import yaml
import argparse
from functools import partial
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


import datasets
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd



# Define constants at the top for easier management
METRICS = [
    'cka_12', 'cka_20', 'cka_28', 'cka_36', 'cka_44', 'cka_52', 'cka_60', 'cka_68', 'cka_76', 'cka_84',
    'cka_92', 'cka_100', 'cknna_10', 'cknna', 'cknna_50', 'mutual_knn_20', 'mutual_knn_40',
    'mutual_knn_60', 'mutual_knn_80', 'mutual_knn_200', 'mutual_knn_400', 'cycle_knn_40',
    'cycle_knn_60', 'cycle_knn_80', 'cycle_knn_100', 'cycle_knn_200', 'cycle_knn_400'
]
ROOT_DIR = "../../data/processed"
SUBJECTS = ['subj01', 'subj02', 'subj05', 'subj07']
OUTPUT_DIR = '../../data/intermediate_results'
LANGUAGE_OUTPUT = os.path.join(OUTPUT_DIR, 'language_all.parquet')
TRAINED_LANGUAGE_OUTPUT = os.path.join(OUTPUT_DIR, 'language_trained.parquet')

VISION_OUTPUT = os.path.join(OUTPUT_DIR, 'vision_all.parquet')
TRAINED_VISION_OUTPUT = os.path.join(OUTPUT_DIR, 'vision_trained.parquet')

# Load configuration once
with open("../../data/utils/Hcp2Yeo7.yaml", 'r') as f:
    HCP2YEO7 = yaml.safe_load(f)


def load_result_file(filename):
    """Load a single result file with error handling"""
    try:
        return torch.load(filename, weights_only=True)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None


def process_subject_results(subject, data_type, metrics):
    """Process results for a single subject and data type (parallelizable function)"""
    print(f"Processing {subject} - {data_type}")
    subject_dir = os.path.join(ROOT_DIR, "comparison", subject, f"{data_type}_fmri")
    results = get_comparison_results(subject_dir, metrics)
    df = dict2df(results)

    # Sort the dataframe
    df = df.sort_values(
        by=['src_model', 'tgt_model', 'metric', 'src_feature', 'tgt_feature']).reset_index(drop=True)

    # Data type specific processing
    if data_type == 'language':
        # Calculate bins for language data
        language_bins = 8
        max_src_feature = df.groupby('src_model')['src_feature'].transform('max')
        df['relative_src_feature'] = df['src_feature'] / (max_src_feature - 1e-3)
        df['relative_src_feature'] = np.floor(df['relative_src_feature'] * language_bins).astype(int)
        df = df[(df['relative_src_feature'] <= language_bins)]
    elif data_type == 'vision':
        # Process vision specific features
        df.loc[:, 'src_feature'] = df['src_feature'].apply(lambda row: int(row.split('_')[-1]))
        vision_bins = 8
        max_src_feature = df.groupby('src_model')['src_feature'].transform('max')
        df['relative_src_feature'] = df['src_feature'] / (max_src_feature + 1)
        df['relative_src_feature'] = np.floor(df['relative_src_feature'] * vision_bins).astype(int)

    return df


def get_comparison_results(root_dir: str, metrics: list[str]):
    """Get comparison results with improved file handling"""
    results = {}

    # First scan directories to avoid redundant checks
    valid_metric_dirs = {}
    for metric in metrics:
        metric_dir = os.path.join(root_dir, metric)
        if os.path.exists(metric_dir):
            valid_metric_dirs[metric] = metric_dir
        else:
            print(f"Metric directory does not exist: {metric_dir}")

    for metric, metric_dir in valid_metric_dirs.items():
        # Get all src directories first
        src_dirs = [d for d in os.listdir(metric_dir)
                    if os.path.isdir(os.path.join(metric_dir, d))]

        # src_dirs = [d for d in src_dirs if 'pythia' not in d]

        for src_path in tqdm(src_dirs, desc=f"Processing {metric}"):
            src_dir = os.path.join(metric_dir, src_path)

            # Get all valid target directories with results files
            tgt_dirs = []
            for tgt_path in os.listdir(src_dir):
                tgt_dir = os.path.join(src_dir, tgt_path)
                if 'pythia' in tgt_dir:
                    step = int(tgt_dir.split('/')[-2].split('_step')[-1])
                    if step not in [0, 1, 2, 4, 8, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
                                    128000]:
                        print(f"[WARNING] Skipping {tgt_dir}")
                        continue

                # if 'pythia' in tgt_path:
                #     continue

                if '---' in tgt_dir:
                    epoch = int(tgt_dir.split('/')[-2].split('---')[-1])
                    if epoch not in [0, 1, 2, 4, 8, 32, 64, 128, 256, 384, 450]:
                        print(f"[WARNING] Skipping {tgt_dir}")
                        continue
                elif '--' in tgt_dir:
                    continue

                if os.path.isdir(tgt_dir) and os.path.exists(os.path.join(tgt_dir, "results.pt")):
                    tgt_dirs.append((tgt_path, tgt_dir))

            # Initialize source entry if needed and valid targets exist
            if tgt_dirs and src_path not in results:
                results[src_path] = {}

            # Process each valid target
            for tgt_path, tgt_dir in tgt_dirs:
                res_filename = os.path.join(tgt_dir, "results.pt")
                res = load_result_file(res_filename)
                if res is None:
                    continue

                # Initialize target entry if needed
                if tgt_path not in results[src_path]:
                    results[src_path][tgt_path] = {
                        'src_filename': res.get('src_filename'),
                        'tgt_filename': res.get('tgt_filename'),
                        'results': {
                            "src_features": res['results'].get('src_features'),
                            "tgt_features": res['results'].get('tgt_features'),
                        }
                    }

                results[src_path][tgt_path]['results'][metric] = res['results'].get('scores')

    return results


def dict2df(results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Convert dictionary to dataframe with vectorized operations where possible"""
    df_rows = []

    for src, tgt_dict in tqdm(results.items()):
        for tgt, res_dict in tgt_dict.items():
            # Extract common data for all rows from this source/target pair
            src_features = res_dict['results'].get('src_features', [])
            tgt_features = res_dict['results'].get('tgt_features', [])

            # Process each metric
            for metric, scores in res_dict['results'].items():
                if metric in ['src_features', 'tgt_features']:
                    continue

                # Create rows in batch for each source-target feature combination
                for src_ids, src_feature in enumerate(src_features):
                    for tgt_ids, tgt_feature in enumerate(tgt_features):
                        # Format features with appropriate prefixes
                        # if 'subj' in src:
                        #     _src_feature = f"{HCP2YEO7[src_feature]}___{src_feature}"
                        # else:
                        #     _src_feature = src_feature
                        #
                        # if 'subj' in tgt:
                        #     _tgt_feature = f"{HCP2YEO7[tgt_feature]}___{tgt_feature}"
                        # else:
                        #     _tgt_feature = tgt_feature

                        df_rows.append({
                            'src_model': src,
                            'tgt_model': tgt,
                            'metric': metric,
                            'src_feature': src_feature,
                            'tgt_feature': tgt_feature,
                            'yeo7': HCP2YEO7[tgt_feature],
                            'score': float(scores[src_ids][tgt_ids]),
                        })

    return pd.DataFrame(df_rows)


def load_benchmark_data(data_type, df):
    """Load benchmark data for either language or vision models"""
    if data_type == 'language':
        # Load Chatbot Arena data
        with open('../../data/utils/elo_results.json', 'r') as f:
            arena_dict = json.load(f)
        arena_df = pd.DataFrame(arena_dict, index=["elo"]).T.reset_index()
        arena_df = arena_df.rename(columns={'index': 'src_model'})
        merged_df = df.merge(arena_df, on='src_model', how='left')

        # Load leaderboard data
        leaderboard = datasets.load_from_disk('../../data/utils/leaderboard.arrow')
        model_names = df['src_model'].unique()

        # Process leaderboard data into a dictionary for faster lookup
        leaderboard_dict = {
            record["fullname"].replace('/', '_'): {
                "Average ⬆️": record["Average ⬆️"],
                "#Params (B)": record["#Params (B)"],
                "IFEval": record["IFEval"],
                "BBH": record["BBH"],
                "MATH Lvl 5": record["MATH Lvl 5"],
                "GPQA": record["GPQA"],
                "MUSR": record["MUSR"],
                "MMLU-PRO": record["MMLU-PRO"],
                "Submission Date": record["Submission Date"],
            }
            for record in leaderboard
            if record["fullname"].replace('/', '_') in model_names
        }

        leaderboard_df = pd.DataFrame(leaderboard_dict).T.reset_index()
        leaderboard_df = leaderboard_df.rename(columns={'index': 'src_model'})
        return merged_df.merge(leaderboard_df, on='src_model', how='left')

    elif data_type == 'vision':
        # Process all vision benchmarks with a more efficient approach
        benchmarks = [
            'imagenet',
            'imagenet-a',
            'imagenet-a-clean',
            'imagenet-r',
            'imagenet-r-clean',
            'imagenet-real',
            'imagenetv2-matched-frequency',
            'sketch'
        ]

        # Get unique source models from the dataframe to avoid unnecessary processing
        src_models = set(df['src_model'].unique())

        # Load all benchmark CSVs at once and build a combined lookup dictionary
        vision_performance_data = {}

        # First collect data from all CSVs
        print("Loading vision benchmark data...")
        for benchmark in tqdm(benchmarks, desc='Loading vision benchmarks'):
            model_csv = pd.read_csv(f'preprocess/extract_vision_features/data/results-{benchmark}.csv')

            # Process each model in the CSV
            for _, row in model_csv.iterrows():
                model_name = row['model']

                # Only process models that exist in our dataframe
                if model_name in src_models:
                    # Initialize model entry if needed
                    if model_name not in vision_performance_data:
                        vision_performance_data[model_name] = {"#Params (M)": 0.0}

                    # Add benchmark score
                    vision_performance_data[model_name][f"{benchmark}"] = 0.01 * row['top1']

                    # Consistently update params (should be the same across benchmarks)
                    vision_performance_data[model_name]["#Params (M)"] = float(row['param_count'].replace(',', ''))

        # Build a consolidated performance dataframe with all benchmarks
        if vision_performance_data:
            print(f"Creating consolidated vision performance dataframe for {len(vision_performance_data)} models...")
            vision_performance_df = pd.DataFrame.from_dict(vision_performance_data, orient='index').reset_index()
            vision_performance_df.rename(columns={'index': 'src_model'}, inplace=True)

            # Single merge operation with the consolidated data
            print("Merging with main dataframe...")
            return df.merge(vision_performance_df, on='src_model', how='left')
        else:
            print("No matching vision benchmark data found.")
            return df
    return None


def ensure_directory_exists(directory):
    """Ensure that a directory exists, creating it if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    # Ensure output directory exists
    ensure_directory_exists(OUTPUT_DIR)

    # Process language data
    print("====== Processing Language Model Comparison Results ======")
    with ProcessPoolExecutor(max_workers=len(SUBJECTS)) as executor:
        # Submit tasks for each subject
        future_to_subject = {
            executor.submit(process_subject_results, subj, 'language', METRICS): subj
            for subj in SUBJECTS
        }

        # Collect results as they complete
        language_dfs = []
        for future in as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                df = future.result()
                language_dfs.append(df)
                print(f"Completed processing language data for {subject}")
            except Exception as e:
                print(f"Error processing language data for {subject}: {e}")

    # Combine all language dataframes
    language_df = pd.concat(language_dfs, ignore_index=True)

    # Load benchmarks and save language data
    merged_language_df = load_benchmark_data('language', language_df)

    pythia_language_df = merged_language_df[merged_language_df['src_model'].str.contains('pythia')]
    merged_language_df = merged_language_df[~merged_language_df['src_model'].str.contains('pythia')]

    pythia_language_df.to_parquet(TRAINED_LANGUAGE_OUTPUT, compression='snappy')
    print(f"Saved language_trained.parquet with {len(pythia_language_df)} rows to {TRAINED_LANGUAGE_OUTPUT}")

    merged_language_df.to_parquet(LANGUAGE_OUTPUT, compression='snappy')
    print(f"Saved language_all.parquet with {len(merged_language_df)} rows to {LANGUAGE_OUTPUT}")

    # Process vision data
    print("====== Processing Vision Model Comparison Results ======")
    with ProcessPoolExecutor(max_workers=len(SUBJECTS)) as executor:
        # Submit tasks for each subject
        future_to_subject = {
            executor.submit(process_subject_results, subj, 'vision', METRICS): subj
            for subj in SUBJECTS
        }

        # Collect results as they complete
        vision_dfs = []
        for future in as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                df = future.result()
                vision_dfs.append(df)
                print(f"Completed processing vision data for {subject}")
            except Exception as e:
                print(f"Error processing vision data for {subject}: {e}")

    # Combine all vision dataframes
    vision_df = pd.concat(vision_dfs, ignore_index=True)

    # Load benchmarks and save vision data
    merged_vision_df = load_benchmark_data('vision', vision_df)

    mixnet_vision_df = merged_vision_df[merged_vision_df['src_model'].str.contains('---')]
    merged_vision_df = merged_vision_df[~merged_vision_df['src_model'].str.contains('---')]

    mixnet_vision_df.to_parquet(TRAINED_VISION_OUTPUT, compression='snappy')
    print(f"Saved vision_trained.parquet with {len(mixnet_vision_df)} rows to {TRAINED_VISION_OUTPUT}")

    merged_vision_df.to_parquet(VISION_OUTPUT, compression='snappy')
    print(f"Saved vision_all.parquet with {len(merged_vision_df)} rows to {VISION_OUTPUT}")


def calc_fmri_alignment():
    ROOT_DIR = "../../data"
    def get_comparison_results(root_dir: str, metrics: list[str]):
        results = {}
        for metric in metrics:
            metric_dir = os.path.join(root_dir, metric)
            if not os.path.exists(metric_dir):
                print(f"Metric directory does not exist: {metric_dir}")
                continue

            for src_path in tqdm(os.listdir(metric_dir), desc=f"Processing {metric}"):
                src_dir = os.path.join(metric_dir, src_path)
                if not os.path.isdir(src_dir):
                    continue  # Skip non-directory files

                if src_path not in results:
                    results[src_path] = {}

                for tgt_path in os.listdir(src_dir):
                    tgt_dir = os.path.join(src_dir, tgt_path)
                    if not os.path.isdir(tgt_dir):
                        continue  # Skip non-directory files

                    res_filename = os.path.join(tgt_dir, "results.pt")
                    if not os.path.exists(res_filename):
                        print(f"Skipping missing file: {res_filename}")
                        continue

                    try:
                        res = torch.load(res_filename, weights_only=True)
                    except Exception as e:
                        print(f"Error loading file {res_filename}: {e}")
                        continue

                    if tgt_path not in results[src_path]:
                        results[src_path][tgt_path] = {
                            'src_filename': res.get('src_filename'),
                            'tgt_filename': res.get('tgt_filename'),
                            'results': {
                                "src_features": res['results'].get('src_features'),
                                "tgt_features": res['results'].get('tgt_features'),
                            }
                        }

                    results[src_path][tgt_path]['results'][metric] = res['results'].get('scores')

        return results

    def dict2df(results: Dict[str, Dict[str, Dict[str, Any]]]):
        df = []
        for src, tgt_dict in tqdm(results.items()):
            for tgt, res_dict in tgt_dict.items():
                for metric, scores in res_dict['results'].items():

                    if metric in ['src_features', 'tgt_features']:
                        continue

                    for src_ids, src_feature in enumerate(res_dict['results']['src_features']):
                        for tgt_ids, tgt_feature in enumerate(res_dict['results']['tgt_features']):
                            yeo7 = hcp2yeo7[src_feature] if 'subj0' in src else hcp2yeo7[tgt_feature]
                            if 'subj' in src:
                                _src_feature = hcp2yeo7[src_feature] + '___' + src_feature
                            else:
                                _src_feature = src_feature
                            if 'subj' in tgt:
                                _tgt_feature = hcp2yeo7[tgt_feature] + '___' + tgt_feature
                            else:
                                _tgt_feature = tgt_feature

                            df.append({
                                'src_model': src,
                                'tgt_model': tgt,
                                'metric': metric,
                                'src_feature': _src_feature,
                                'tgt_feature': _tgt_feature,
                                # 'yeo7': hcp2yeo7[tgt_feature],
                                'score': float(scores[src_ids][tgt_ids]),
                            })

        return pd.DataFrame(df)

    fmri_dir = os.path.join(f"{ROOT_DIR}/processed", "comparison", "subj01", "fmri_fmri")
    hcp2yeo7 = yaml.safe_load(open(f"{ROOT_DIR}/utils/Hcp2Yeo7.yaml"))

    fmri_res = get_comparison_results(fmri_dir, ['cka_36'])
    fmri_df = dict2df(fmri_res)

    fmri_df = fmri_df.sort_values(by=['src_model', 'tgt_model', 'metric', 'src_feature', 'tgt_feature']).reset_index(
        drop=True)

    fmri_df['src_yeo7'] = fmri_df['src_feature'].apply(lambda x: x.split('___')[0])
    fmri_df['src_feature'] = fmri_df['src_feature'].apply(lambda x: x.split('___')[1])

    fmri_df['tgt_yeo7'] = fmri_df['tgt_feature'].apply(lambda x: x.split('___')[0])
    fmri_df['tgt_feature'] = fmri_df['tgt_feature'].apply(lambda x: x.split('___')[1])

    fmri_df.to_parquet(f'{OUTPUT_DIR}/fmri_alignment.parquet', compression='snappy')

if __name__ == '__main__':
    main()
    calc_fmri_alignment()