# Interpretable Detection of Anatomical Change for Adaptive Radiotherapy (ART)

This repository contains the code and data used to develop and evaluate a simple, interpretable workflow for detecting clinically relevant anatomical change from daily imaging in head and neck radiotherapy. The pipeline consists of:

1. Leave-one-out cross-validation (LOOCV) model training and evaluation  
2. Notebooks for figure generation and data analysis  



## Setup

### 1. Create the Conda environment

From the repository root:

    conda env create -f environment.yml
    conda activate ART_Volumetric

### 2. Data

Ensure the file `data/all_data.pkl` is available.  
This contains the per-patient volumetric changes over the treatment fractions.



## Running LOOCV Training & Evaluation

### Local execution

    python scripts/training.py

This will:

- Load the dataset  
- Run leave-one-out cross-validation  
- Train and evaluate all feature-map / hyperparameter variants  
- Write results to `results/`:
  - `loo_all_fold_preds.pkl`
  - `loo_selection_results.csv`
  - diagnostic plots and `training.log`

### Running with SLURM

    sbatch training.sbatch

SLURM output will be stored in `logs/slurm/`.



## Notebooks for Analysis & Figure Generation


### `notebooks/daily_analysis.ipynb`

- Computes volumetric trends  
- Identifies significant differences between replanned and non-replanned groups  
- Generates the volume-difference figure

### `notebooks/loocv_analysis.ipynb`

- Loads LOOCV outputs  
- Summarizes performance (recall, F1-score, specificity, early detection rate)    

Exported figures are saved into `figures/`.


