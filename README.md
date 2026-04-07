# Navigating the Rugged Landscape: A Robust Gradient-Guided Framework for Imperceptible 3D Adversarial Point Clouds

This repository contains the official implementation of our paper:

**Navigating the Rugged Landscape: A Robust Gradient-Guided Framework for Imperceptible 3D Adversarial Point Clouds**


## Overview

We propose a robust gradient-guided framework for generating imperceptible adversarial examples on 3D point clouds. The framework is designed to mitigate gradient instability and myopic updates under rugged 3D loss landscapes.

Our method consists of three key components:

- **Gaussian Gradient Smoothing (GGS):** improves gradient robustness by aggregating neighboring gradients.
- **Cyclic Gradient Calibration (CGC):** refines the attack trajectory through cyclic lookahead-based calibration.
- **Normalized Gradient Strategy (NGS):** preserves directional fidelity during optimization.

The proposed framework can be integrated into existing gradient-based 3D adversarial attacks to improve adversarial imperceptibility.

## Datasets

The test dataset used in our experiments can be downloaded from the following link:

- **Test dataset:** [Google Drive](https://drive.google.com/drive/folders/1NKOletCy8dsdLRTS21xXdvSJjo_F6MOZ?usp=drive_link)

After downloading, please place the dataset files in the `data/` directory.

## Trained Models

The trained models used in our experiments can be downloaded from the following link:

- **Trained models:** [Google Drive](https://drive.google.com/drive/folders/1ahpl0_y05t4IjsKT1Gk8t4WR6rha8yD1?usp=drive_link)

After downloading, please place the checkpoints in the `log/` directory.

## Main Arguments

- `--strategy`: attack strategy.  
  - `Default`: the baseline attack method.  
  - `GGS_CGC_NGU`: the proposed framework.

- `--dataset`: dataset used for evaluation.  
  - `ModelNet`: ModelNet40 dataset.  
  - `ShapeNetPart`: ShapeNet Part dataset.

- `--log_dir`: victim model / checkpoint directory.  
  - `pointnet`: PointNet  
  - `dgcnn`: DGCNN  
  - `pointnet2_msg`: PointNet++  
  - `curvenet_cls`: CurveNet
