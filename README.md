This is the project page of our paper:
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

