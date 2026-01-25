# Baseline-Registration

**ANTs/Simple ITK**

This repository provides an implementation of **ANTs-affine**, **ANTs-SyN**, and **SITK-Bspline**.  
The code targets **same-modality X-ray registration** and supports forward–backward registration with **inverse consistency error (ICE)** evaluation.

## Features

- Classical intensity-based registration backends (ANTs / SimpleITK)
- Forward–backward registration for ICE evaluation
- Metrics: SSIM, NMI, ICE, TV, Jacobian determinant statistics
- Visualization of warped images and deformation fields

## Repository Structure

text
.
├─ ants_reg.py
├─ sitk_reg.py
└─ test_data

## Installation

1. Clone the repo

"```bash
git clone https://github.com/CHENG-EN-TSAI/Baseline-Registration.git
```"

2. Create new conda environment

"```bash
conda create -n reg python==3.9
conda activate reg
```"

3. Install required packages

"```bash
pip install -r requirements.txt
```"

## Navigate to the project directory

"```bash
cd Baseline-Registration
```"

## Data Preparation

`--data_paths_json` must point to a JSON file containing a **list of dictionaries**.  
Each dictionary specifies one registration pair.

**Required fields**
- `fixed`: path to the fixed image
- `moving`: path to the moving image

### Example JSON format

"```json
[
  {
    \"fixed\": \"FIXED_IMAGE_PATH\",
    \"moving\": \"MOVING_IMAGE_PATH\"
  }
]
```"

A sample JSON file is provided in `test_data/` for testing purposes.

For **Medalab members**, create a symbolic link to the full NIH ChestX-ray8 dataset as follows.  
The corresponding JSON file is located in this directory:

"```bash
ln -s /data2/smarted/TMUH/data data
```"

**Note:** Please contact me if you need to modify anything in this directory.

## RUN a Test

### Notes
- For reproducibility, set `reproducible = True`

Run ANTs-Affine:

"```bash
python ants_reg.py --method Affine
```"

Run ANTs-SyN:

"```bash
python ants_reg.py --method SyN
```"

Run SITK-Bspline:

"```bash
python sitk_reg.py
```"

## Arguments

text

## Output

Results are saved to `{result_path}/{i}/` including `metrics.json` and visualizations.  
After completion, mean ± std of **SSIM, NMI, ICE, TV_mean, TV_p95, DJ_fold%, DJ_p95, DJ_p5, and MSE[D_J-c]** are printed.

## Citation

## Acknowledgements

ANTs/SITK toolkits, NIH Chest X-ray dataset

## Contact

If you have any questions, please contact f200154nn6@gmail.com
