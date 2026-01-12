# Fractional Fourier Transform–Based Deep Learning for Surface-Wave Inversion

This repository contains code for inverting S-wave velocity profiles from surface waves. The code implements and evaluates a fractional Fourier transform (FrFT)–based dispersion representation combined with a convolutional neural network (CNN) for shear-wave velocity (Vs) inversion, and compares its performance against the conventional Fourier transform (FFT).

### **Repository Contents**

The repository contains three main Python scripts corresponding to different components of the study:

 1. `train_fft_frft_cnn.py`

Purpose:

Training and validation of the CNN using FFT and multiple FrFT representations of dispersion data.

Main functionalities:

Loads dispersion and Vs datasets

Applies FFT and FrFT with different fractional orders

Trains an identical CNN architecture for all representations

Compares validation loss across transforms

Identifies optimal fractional order based on minimum MSE


2. `frft_dispersion_visualization.py`

Purpose:

Visualization of dispersion curves in the fractional Fourier domain.

Main functionalities:

Applies discrete FrFT to phase and group velocity curves

Generates dispersion representations for selected rotation angles

Illustrates the transition from physical to fractional domains


3. `noise_sensitivity_analysis.py`

Purpose:

Quantitative noise robustness analysis of FFT- and FrFT-based representations.

Main functionalities:

Trains CNN models using noise-free data

Injects controlled Gaussian noise at different SNR levels during testing

Evaluates inversion performance using mean squared error (MSE)

Compares noise sensitivity across FFT and FrFT orders

### **Dataset**

The dispersion and velocity model datasets are obtained from:

Zenodo Repository:
[https://github.com/liufeng2317/DispFormer/tree/master](https://zenodo.org/records/14619577)

Files required:

`period_phase_group.npz` — dispersion data (period, phase velocity, group velocity)

`depth_vs.npz` — corresponding shear-wave velocity profiles

Place these files in the root directory before running the scripts.

Requirements

The code was tested using Python ≥ 3.8 with the following packages:
```text
numpy
scipy
matplotlib
scikit-learn
torch
```

Install dependencies using:
```text
pip install numpy scipy matplotlib scikit-learn torch
```

### **Reproducibility Notes**

- All CNN architectures, training strategies, and hyperparameters are kept identical across FFT and FrFT experiments.

- FrFT-based representations are used only as input transformations; no architectural modifications are introduced.

- Noise is added only during testing for noise sensitivity analysis, ensuring fair robustness evaluation.

- Random seeds can be fixed for exact reproducibility if required.

### **Important Note on Physical Interpretation**

The FrFT-based dispersion representations used in this repository are not intended to preserve the physical meaning of phase and group velocities.
Instead, they provide an alternative signal-domain representation that redistributes dispersion energy and improves feature learning for data-driven inversion.

