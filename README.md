# Speech MRI segmentation
This repository contains Python code to perform deep-learning-based segmentation of two-dimensional (2D) magnetic resonance (MR) images of the vocal tract. Code to train segmentation convolutional neural networks (CNNs) from scratch is included, as well as code to estimate segmentations using trained segmentation CNNs. 

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
6. Enter the following command to check that the Barts Speech MRI dataset is correctly organised and is not corrupted:
```
python CheckData.py
```
By default, CheckData.py assumes that the folder containing the dataset is called *data*, and that the data of subjects 1, 2, 3, 4 and 5 should be checked. However, these defaults can be overridden using the following arguments:
```
python CheckData.py --data_dir /path/to/folder --subj_id_list [1,2,4]
```
In the example above, only the data of subjects 1, 2 and 4 would be checked.

7. Enter the following command to normalise the images of the Barts Speech MRI dataset:
```
python NormaliseImages.py
```
By default, NormaliseImages.py assumes that the folder containing the dataset is called *data*, and that the data of subjects 1, 2, 3, 4 and 5 should be normalised. However, these defaults can be overridden using the following arguments:
```
python NormaliseImages.py --data_dir /path/to/folder --subj_id_list [1,2,4]
```

## Training a segmentation CNN from scratch
Enter the following command to train a CNN to segment 2D MR images of the vocal tract from scratch:
```
python TrainCNN.py
```
By default, TrainCNN.py makes the following assumptions:
- Name of folder containing the entire dataset: *data*
- Subjects to include in training dataset: 2, 3 and 4
- Subject to include in validation dataset: 1
- Number of segmentation classes (including background): 7
- Number of epochs of training: 200
- Learning rate to use in training: 0.0003
- Mini-batch size to use in training: 4
- ID of GPU to use in training: 0
However, these assumptions can be overridden using the following arguments:
```
python TrainCNN.py --data_dir /path/to/folder --train_subj [1,3] --val_subj [2,4] --n_classes 5 --epochs 10 --l_rate 0.01 --mb_size 8 --gpu_id 1
```
In the example above:
- Subjects to include in training dataset: 1 and 3
- Subject to include in validation dataset: 2 and 4
- Number of segmentation classes (including background): 5
- Number of epochs of training: 10
- Learning rate to use in training: 0.01
- Mini-batch size to use in training: 8
- ID of GPU to use in training: 1
