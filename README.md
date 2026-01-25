# Baseline-Registration

**ANTs / SimpleITK**

This repository provides baseline implementations of classical medical image registration methods:

- **ANTs-Affine**
- **ANTs-SyN**
- **SimpleITK Affine + BSpline**

The code targets **same-modality X-ray registration** and supports **forward–backward registration** with **Inverse Consistency Error (ICE)** evaluation.

---

## Features

- Registration backends
  - ANTs: `Affine`, `SyN`
  - SimpleITK: two-stage `Affine → BSpline`
- Forward + backward registration for ICE
- Metrics (per pair)
  - SSIM
  - NMI
  - ICE
  - TV (Total Variation) of the displacement field
  - Jacobian determinant statistics: folding %, p5, p95
  - Optional: `MSE[D_J - c]` when metadata + segmentation are available
- Outputs and visualizations
  - `metrics.json`
  - Warped image and GIF animation
  - Deformation grid visualization
  - Deformation vector field plot
  - (SITK pipeline) folding overlay visualization if enabled in the script

---

## Repository Structure

```text
.
├─ ants_reg.py
├─ sitk_reg.py
└─ test_data
   └─ test.json
