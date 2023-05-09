# EvaluateCNN.py
# Code to evaluate a U-Net model to segment
# articulators and the vocal tract in real-time
# two-dimensional magnetic resonance images of the vocal 
# tract during speech

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 9th May 2023

# Import required modules
import os
import argparse
from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from DataAugmentation import ToTensor, RotateCropAndPad, RandomCrop, Rescale, RandomTranslation, RescaleAndPad
from SpeechMRIDataset import SpeechMRIDataset
from UNet_n_classes import UNet_n_classes
import pandas as pd
from monai.metrics import DiceMetric
from TrainCNN import create_dataset


def main(data_dir, test_subj, cnn_dir, n_classes):

    """Function performs the following steps:
       1) Loads images for testing
       2) Sets up the U-Net model
       3) Tests the model
       4) Saves estimated segmentations"""

    # Create test dataset
    test_dataset, loss_weighting = create_dataset(data_dir, test_subj, n_classes, augmentation=False)

    # Print update on test dataset
    print(f'Test dataset created consisting of images of subject(s) {test_subj}')

    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)
    
    # Specify GPU where training will occur
    device = torch.device("cuda:0")

    # Function to calculate Dice coefficients of estimated segmentations
    calc_dsc = DiceMetric(include_background=False, reduction='none')

    # Create string of test dataset subject IDs
    test_subj_string = [str(f) for f in test_subj]
    test_subj_string = "_".join(test_subj_string)
    
    # Set up the U-Net model
    unet = UNet_n_classes(n_classes)

    # Load U-Net model parameters
    unet.load_state_dict(torch.load(cnn_dir / 'unet_parameters.pth'))
    
    # Send model to GPU
    unet = unet.to(device)

    # Set model to evaluation mode
    unet.eval()
    with torch.no_grad():

        # For each image in test dataset
        for speech_data in test_dataloader:

            # Extract inputs to model and labels
            unet_inputs = speech_data['image_frame'].to(device)
            labels = speech_data['mask_frame'].to(device)

            # Forward propagation
            unet_outputs = unet(unet_inputs)

            # Estimated segmentations
            est_segs = torch.argmax(unet_outputs.data, dim=1)

            # One-hot encode segmentations
            tmp_est_segs = F.one_hot(est_segs, num_classes=n_classes)
            labels = F.one_hot(labels, num_classes=n_classes)

            # Permute dimensions of segmentations
            tmp_est_segs = torch.permute(tmp_est_segs, (0, 3, 1, 2))
            labels = torch.permute(labels, (0, 3, 1, 2))

            # Calculate the Dice coefficient
            dsc = calc_dsc(tmp_est_segs, labels)

            # Create pandas DataFrame of Dice coefficients
            df = pd.DataFrame(dsc.cpu())

            # Save df
            df.to_csv(cnn_dir / f'test_subj_{test_subj_string}_dsc_values.csv', index=False)

            # Print update
            print(f'Dice coefficients of segmentations estimated by segmentation CNN saved here: "{cnn_dir / f"test_subj_{test_subj_string}_dsc_values.csv"}"')

            # Save segmentations estimated by segmentation CNN
            savemat(cnn_dir / f'test_subj_{test_subj_string}_estimated_segmentations.mat', {'est_segs': np.uint8(torch.permute(est_segs, (1, 2, 0)).cpu().numpy())})

            # Print update
            print(f'Segmentations estimated by segmentation CNN saved here: "{cnn_dir / f"test_subj_{test_subj_string}_estimated_segmentations.mat"}"')

    
if __name__ == "__main__":
  
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Code to evaluate a convolutional neural network for image segmentation.')
    
    # Add arguments
    parser.add_argument(
        '--data_dir', 
        help='The path to the folder containing all the data.',
        default='data',
        type=Path
        )
    parser.add_argument(
        '--test_subj',
        help='A list of ID(s) of subjects to use in the testing dataset.',
        type=int,
        default=1,
        nargs='*'
        )
    parser.add_argument(
        '--cnn_dir',
        help='The path to the folder containing the CNN weights.',
        type=Path,
        default='data/SegmentationCNNs/val_subj_1_train_subj_2_3_4_l_rate_0.0003_mb_size_4_epochs_10'
        )
    parser.add_argument(
        '--n_classes',
        help='Number of classes (including background) in ground-truth segmentations.',
        type=int,
        default=7
        )
    parser.add_argument(
        '--gpu_id',
        help='ID of GPU to use for training.',
        type=int,
        default=0
        )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if CUDA is available
    assert torch.cuda.is_available(), 'PyTorch does not detect any GPUs.'

    # Select GPU to use for training
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Check if data_dir exists
    assert os.path.exists(args.data_dir), 'Please specify the absolute path to the folder containing all the data using the --data_dir argument to "TrainCNN.py".'

    # Check if cnn_dir exists
    assert os.path.exists(args.cnn_dir), 'Please specify the absolute path to the folder containing the CNN weights using the --cnn_dir argument to "TrainCNN.py".'

    # Check if images have been normalised
    assert os.path.exists(args.data_dir / 'Normalised_Images'), 'Have the images been normalised? This can be done using "NormaliseImages.py".'
    
    # Check that args.test_subj is not empty
    assert args.test_subj, f'Please specify the IDs of the subjects whose datasets should be included in the testing dataset using the --test_subj argument to "EvaluateCNN.py".'
    
    # If required, modify args.test_subj
    if isinstance(args.test_subj, int):
        args.test_subj = [args.test_subj]

    # Run main function
    main(args.data_dir, args.test_subj, args.cnn_dir, args.n_classes)