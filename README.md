# Speech MRI segmentation
This repository contains Python code to perform deep-learning-based segmentation of two-dimensional magnetic resonance images of the vocal tract. Code to train segmentation convolutional neural networks (CNNs) from scratch is included, as well as code to estimate segmentations using trained segmentation CNNs. 

## Introduction



## Requirements

1. Software: [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Hardware: Computer with one or more graphical processing units (GPUs) with memory > GB

## Setting up

1. Download this repository.

2. Open a new terminal and navigate to the folder containing the files from this repository.

3. Enter the following command to create a conda environment and install the Python packages required to run the code:
```
conda env create -f environment.yml
```
4. Enter the following command to activate the conda environment
```
conda activate SpeechMRISeg
```
5. If using the Barts Speech MRI dataset, add the dataset to the folder containing the files from this repository:
```
.
├── data
│   ├── GT_Segmentations
│   ├── MRI_SSFP_10fps
│   └── Velopharyngeal_Closure
├── CheckData.py
├── environment.yml
├── NormalisedImages.py
├── README.md
├── SpeechMRIDataset.py
├── TrainCNN.py
└── UNet_n_classes.py
```
