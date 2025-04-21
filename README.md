# Brain-AI Alignment: Convergent Evolution Across Modalities, Scales, and Training Trajectories

This repository contains the official implementation for the paper "Alignment Between Brains and AI: Evidence for Convergent Evolution Across Modalities, Scales and Training Trajectories". 

![overview](./figures/overview.png)

## Overview

Our study investigates how artificial neural networks and biological brains may independently develop similar representational strategies by analyzing alignment between AI model internal representations and human brain activity. We performed a large-scale analysis across hundreds of models spanning diverse architectures, scales, training trajectories, and two sensory modalities (vision and language), generating over 60 million distinct alignment measurements.

Key findings:
- Higher-performing models spontaneously develop stronger brain alignment without explicit neural constraints
- This effect is stronger in language models (r=0.89) than vision models (r=0.56)
- Increased brain alignment consistently precedes performance improvements during training
- Alignment patterns shift systematically from posterior to anterior brain regions as representational scope moves from local to global features

## Repository Structure

```
.
├── code
│   ├── alignment/               # Code for computing brain-AI alignment
│   ├── analysis/                # Analysis and visualization scripts
│   ├── brain_datasets.py        # Utilities for handling brain data
│   ├── feature_extraction/      # Extract representations from models
│   │   ├── language/            # Language model feature extraction
│   │   ├── vision/              # Vision model feature extraction
│   │   │   └── vision_model_training/  # Scripts for training vision models
│   │   └── models.yaml          # Model configurations
│   ├── preprocessing/           # Scripts for data preparation 
│   └── requirements.txt         # Python dependencies
├── data
│   ├── intermediate_results/    # Precomputed alignment results (included)
│   └── utils/                   # Utility scripts for data handling
├── figures
│   ├── brain_regions/           # Brain region visualization figures
│   ├── kernel_size/             # Kernel size analysis figures
│   ├── layer_wise/              # Layer-wise alignment figures
│   ├── overview/                # Overview and methodology figures
│   ├── performance/             # Performance-alignment correlation figures 
│   ├── scatter/                 # Scatter plot visualizations
│   ├── statistic/               # Statistical analysis figures
│   └── training/                # Training trajectory figures
└── README.md
```

## Installation Requirements

### Hardware Requirements

- Linux operating system
- CUDA-compatible GPUs (recommended for faster computation)
- Storage space for the Natural Scenes Dataset (NSD) and model weights

### Software Requirements

1. Install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) (required for brain data analysis)

2. Set up the Python environment:
    ```bash
    cd code
    pip install -r requirements.txt
    ```

## Data Preparation

There are two options for reproducing the results:
1. Run the entire pipeline from scratch
2. Use our precomputed alignment results

### Option 1: Running the Full Pipeline

1. Download the Natural Scenes Dataset (NSD) from [https://naturalscenesdataset.org](https://naturalscenesdataset.org)
2. Place the data in the `data` directory with the following structure:
```
./data
├── nsddata/
├── nsddata_betas/
├── nsddata_stimuli/
└── processed/
```

### Option 2: Using Precomputed Results

Our precomputed alignment results are already included in the `data/intermediate_results/` directory, containing:
- `fmri_alignment.parquet`: Alignment scores for different subjects and brain regions
- `language_all.parquet`: Language model layers' alignment scores and benchmark results 
- `vision_all.parquet`: Vision model layers' alignment scores and benchmark results
- `language_trained.parquet`: Training checkpoints for Pythia model family
- `vision_trained.parquet`: Training checkpoints for MixNet model family

You can directly proceed to the [Results Analysis](#results-analysis) section if you want to use these precomputed results.

## Preprocessing

### Extracting Image Stimuli and Captions

    ```bash
    cd code/preprocessing
    python extract_img_caps.py
    ```

### Preparing Brain Activity Data

    ```bash
    cd code/preprocessing
    python make_subjmri.py --subj subj01
    python make_subjmri.py --subj subj02
    python make_subjmri.py --subj subj05
    python make_subjmri.py --subj subj07
    python fmri2pt.py
    ```

## Feature Extraction

### Language Models

1. Modify model paths in `./code/feature_extraction/models.yaml`
2. Download models from Hugging Face (recommended)
3. Extract features:
    ```bash
    cd code/feature_extraction/language
    python extract_language_feature_all.py --subj subj01
    python extract_language_feature_all.py --subj subj02
    python extract_language_feature_all.py --subj subj05
    python extract_language_feature_all.py --subj subj07
    ```

For single model extraction:
    ```bash
    python extract_language_feature_single.py --help
    ```

### Vision Models

1. Extract features for pre-trained models:
    ```bash
    cd code/feature_extraction/vision
    python extract_vision_feature_all.py --subj subj01
    python extract_vision_feature_all.py --subj subj02
    python extract_vision_feature_all.py --subj subj05
    python extract_vision_feature_all.py --subj subj07
    ```

   2. For training trajectory analysis (optional):
      - Download ImageNet from [http://www.image-net.org/download](http://www.image-net.org/download)
      - Train MixNet models:
        ```bash
        cd code/feature_extraction/vision/vision_model_training
        torchrun --master_port 29500 --nproc_per_node=4 train.py \
            --data-dir /path/to/ILSVRC2012/ \
            --model mixnet_s \
            -b 128 \
            --sched step \
            --epochs 450 \
            --decay-epochs 2.4 \
            --decay-rate .969 \
            --opt rmsproptf \
            --opt-eps .001 \
            -j 8 \
            --warmup-lr 1e-6 \
            --weight-decay 1e-5 \
            --drop 0.3 \
            --drop-path 0.2 \
            --model-ema \
            --model-ema-decay 0.9999 \
            --aa rand-m9-mstd0.5 \
            --remode pixel \
            --reprob 0.3 \
            --amp \
            --lr .016 \
            --dist-bn reduce \
            --checkpoint-hist 450
        ```

3. Evaluate model performance:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python validate.py \
     --data-dir /path/to/ILSVRC2012/ \
     --model mixnet_s \
     -b 128 \
     -j 8 \
     --checkpoint /path/to/checkpoint.pth.tar
   ```

## Computing Alignment

Run the alignment computation:
    ```bash
    cd code/alignment
    bash align.sh
    ```

To customize GPU utilization, modify the `--tasks-per-gpu` parameter in `align.py`.

After alignment computation, collect results:
    ```bash
    python results2parquet.py
    ```

## Results Analysis

The analysis scripts are located in `code/analysis/`. You need to modify `ROOT_DIR` in each script to point to your project root directory.

### Analysis Notebooks

- `analysis.ipynb`: General analysis and visualization for Figures 1d-e, 2a-b, 4a, 5, and 6
- `analysis_model_training.ipynb`: Training trajectory analysis for Figure 3a-f
- `umap_visualization.ipynb`: UMAP visualization for Figure 2c
- `surface_visualization.py`: Brain surface visualization for Figure 4b


<!-- ## Citation

If you find this work useful, please cite our paper:
    ```
    TBD
    ``` -->

## Acknowledgments

This research utilized the Natural Scenes Dataset (NSD), which contains fMRI recordings from multiple subjects viewing thousands of naturalistic images from the COCO dataset. We thank the NSD team for making this data publicly available.

Our analysis included models from various sources:
- Language models: Qwen, Llama, Gemma, Mistral, Phi, and others via HuggingFace
- Vision models: Various architectures from the TIMM library

<!-- ## License


[Specify license information]


## Contact


For questions or issues, please open an issue in this repository or contact [email](Floyed_Shen@outlook.com). -->