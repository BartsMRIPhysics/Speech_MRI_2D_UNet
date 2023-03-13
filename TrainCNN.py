# TrainCNN.py
# Code to train a U-Net model to segment
# articulators and the vocal tract in real-time
# two-dimensional magnetic resonance images of the vocal 
# tract during speech

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 13th March 2023

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


def create_dataset(data_dir, subj_id_list, n_classes, augmentation=True):
    """Function to create a dataset. 

    Args:
        - data_dir (Path): path to folder containing all data (i.e. images
          and ground-truth segmentations)
        - subj_id_list (list of integers): list of IDs of subjects 
          whose images should be included in dataset.
          For example, subj_id_list = [1,2,3,4] would indicate
          that the images of subjects 1, 2, 3 and 4 should be included
          in the dataset
        - n_classes (integer): number of classes (including background)
          in ground-truth segmentations
        - augmentation (True or False): indicates if data should be 
          augmented using rotations, translations, cropping and rescaling
        
    Returns:
        
        - PyTorch dataset
        - loss_weighting (PyTorch tensor): the weighting of each class in 
          the loss function
    """
    
    # Preallocate list for paths to images and corresponding 
    # ground-truth (GT) segmentations to include in dataset
    full_img_list = []
    full_seg_list = []

    # For each subject ID
    for subj_id in subj_id_list:

        # Create lists of frames in the subfolders
        img_list = [int(g[6:-4]) for g in os.listdir(data_dir / 'Normalised_Images' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('image'))]
        img_list.sort()
        seg_list = [int(g[5:-4]) for g in os.listdir(data_dir / 'GT_Segmentations' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('mask'))]
        seg_list.sort()

        # Print update
        print(f'Subject {subj_id} has {len(img_list)} images and corresponding ground-truth segmentations')

        # Create lists of paths to images and GT segmentations
        img_list = [data_dir / 'Normalised_Images' / f'Subject{subj_id}' / f'image_{g}.mat' for g in img_list]
        seg_list = [data_dir / 'GT_Segmentations' / f'Subject{subj_id}' / f'mask_{g}.mat' for g in seg_list]

        # Update full_img_list and full_seg_list
        full_img_list += img_list
        full_seg_list += seg_list

    # Load first image
    frame = loadmat(full_img_list[0])['image_frame']

    # Image dimensions
    first_frame_dim = frame.shape

    # Preallocate PyTorch tensor to calculate number of pixels per class
    pix_per_class = torch.zeros((first_frame_dim[0], first_frame_dim[1], n_classes))

    # For each segmentation
    for file_path in full_seg_list:

        # Load segmentation
        frame = loadmat(file_path)['mask_frame'].astype('int64')

        # Calculate number of pixels per class
        tmp_pix_per_class = torch.from_numpy(frame)
        tmp_pix_per_class = F.one_hot(tmp_pix_per_class)

        # Update pix_per_class
        pix_per_class += tmp_pix_per_class

    # Calculate number of pixels per class
    pix_per_class = torch.sum(pix_per_class, (0, 1))

    # Calculate loss weighting of each class
    loss_weighting = torch.sum(pix_per_class) / pix_per_class

    # If required, augment the dataset
    if augmentation:

        # Create dataset with transformation
        data_transform = transforms.Compose([ToTensor()])
        dataset_1 = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

        # Create dataset with transformation
        data_transform = transforms.Compose([RotateCropAndPad((-30, 10)), ToTensor()])
        dataset_2 = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

        # Create dataset with transformation
        data_transform = transforms.Compose([RandomCrop((15, 30), (5, 10), 220), Rescale(256), ToTensor()])
        dataset_3 = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

        # Create dataset with transformation 
        data_transform = transforms.Compose([RandomTranslation((-30, 30), (0, 30)), ToTensor()])
        dataset_4 = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

        # Create dataset with transformation 
        data_transform = transforms.Compose([RescaleAndPad((210, 255)), ToTensor()])
        dataset_5 = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

        # Combine datasets
        dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5])

    else:
        
        # Define transformation
        data_transform = transforms.Compose([ToTensor()])

        # Create training, validating and testing datasets
        dataset = SpeechMRIDataset(full_img_list, full_seg_list, transform = data_transform)

    return dataset, loss_weighting


def main(data_dir, train_subj, val_subj, n_classes, epochs, l_rate, mb_size):

    """Function performs the following steps:
       1) Loads images for training (and validation)
       2) Sets up the U-Net model
       3) Trains (and validates) the model
       4) Saves model parameters and training (and validation) losses"""

    # Create training dataset
    training_dataset, loss_weighting = create_dataset(data_dir, train_subj, n_classes, augmentation=True)

    # Print update on training dataset
    print(f'Training dataset created consisting of images of subject(s) {train_subj}')

    # If required, create a validation dataset
    if val_subj:
        validation_dataset, loss_weighting = create_dataset(data_dir, val_subj, n_classes, augmentation=False)
        
        # Print update on validation dataset
        print(f'Validation dataset created consisting of images of subject(s) {val_subj}')
    else:
        # Print update on validation dataset
        print(f'No validation dataset')

    # Create dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=mb_size, shuffle=True, num_workers=4)
    if val_subj:
        validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=4)

    # Specify GPU where training will occur
    device = torch.device("cuda:0")

    # Send loss_weighting to GPU
    loss_weighting = loss_weighting.to(device)
    
    # Set up the U-Net model
    unet = UNet_n_classes(n_classes)
    
    # Send model to GPU
    unet = unet.to(device)

    # Zero the gradient buffers
    unet.zero_grad()

    # Create the optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=l_rate)

    # Create the loss function
    mean_loss_func = nn.CrossEntropyLoss(weight=loss_weighting)

    # Function to calculate Dice coefficients of estimated segmentations
    calc_dsc = DiceMetric(include_background=False, reduction='none')

    # Preallocate lists for training (and validation) losses (and validation Dice coefficients)
    train_loss_list = []
    if val_subj:
        val_loss_list = []
        mean_dsc_list = []
        mean_dsc_per_class = torch.zeros((n_classes - 1, epochs), dtype=torch.float, device=device)

    # Print updates
    print('Training of segmentation CNN started')
    print(f'Training duration: {epochs} epochs')
    print(f'Learning rate: {l_rate}')
    print(f'Mini-batch size: {mb_size}')

    # Train the U-Net model
    for epoch in range(1, epochs + 1):
    
        # Training loss
        training_loss = 0.

        # Set to model to training mode
        unet.train()

        # For each mini-batch in training dataset
        for idx, speech_data in enumerate(training_dataloader):
        
            # Extract inputs to model and labels
            unet_inputs = speech_data['image_frame'].to(device)
            labels = speech_data['mask_frame'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward propagation
            unet_outputs = unet(unet_inputs)

            # Calculate loss
            unet_loss = mean_loss_func(unet_outputs, labels)

            # Backwards propagation
            unet_loss.backward()

            # Optimise weights
            optimizer.step()

            # Update training loss
            training_loss += unet_loss.item()

        # Calculate mean training loss and update train_loss_list
        tmp_training_loss = training_loss / len(training_dataloader)
        train_loss_list.append(tmp_training_loss)

        # If required, calculate validation loss
        if val_subj:
            
            # Set model to evaluation mode
            unet.eval()
            with torch.no_grad():

                # For each image in validation dataset
                for speech_data in validation_dataloader:

                    # Extract inputs to model and labels
                    unet_inputs = speech_data['image_frame'].to(device)
                    labels = speech_data['mask_frame'].to(device)

                    # Forward propagation
                    unet_outputs = unet(unet_inputs)

                    # Calculate loss
                    validation_loss = mean_loss_func(unet_outputs, labels)

                    # Update val_loss_list
                    val_loss_list.append(validation_loss.item())

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

                    # Update mean_dsc_per_class
                    mean_dsc_per_class[:, epoch - 1] = torch.mean(dsc, dim=0)

                    # Update mean_dsc_list
                    dsc = torch.mean(dsc).item()
                    mean_dsc_list.append(dsc)

        # Print epoch and loss information
        if val_subj:
            print(f'Epoch: {epoch}   Training loss: {tmp_training_loss:.4f}   Validation loss: {validation_loss.item():.4f}   Dice coefficient: {dsc:.4f}')
        else:
            print(f'Epoch: {epoch}   Training loss: {tmp_training_loss:.4f}')
    
    # Print update
    print('Finished segmentation CNN training')

    # Create string of training dataset subject IDs
    train_subj_string = [str(f) for f in train_subj]
    train_subj_string = "_".join(train_subj_string)

    # Path to folder where parameters of trained model will be saved
    if val_subj:

        # Create string of training dataset subject IDs
        val_subj_string = [str(f) for f in val_subj]
        val_subj_string = "_".join(val_subj_string)

        # Path to folder
        save_dir_path = data_dir / 'SegmentationCNNs' / f'val_subj_{val_subj_string}_train_subj_{train_subj_string}_l_rate_{l_rate}_mb_size_{mb_size}_epochs_{epochs}'    
    else:
        save_dir_path = data_dir / 'SegmentationCNNs' / f'train_subj_{"_".join(train_subj)}_l_rate_{l_rate}_mb_size_{mb_size}_epochs_{epochs}'
    
    # If required, create folders
    if os.path.exists(save_dir_path):
        # Print update
        print(f'{save_dir_path} already exists')
    else:
        os.makedirs(save_dir_path)
        # Print update
        print(f'{save_dir_path} created')
    
    # Save parameters of trained model
    torch.save(unet.state_dict(), save_dir_path / 'unet_parameters.pth')

    # Print update
    print(f'Parameters of trained segmentation CNN saved here: "{save_dir_path / "unet_parameters.pth"}"')

    # Create a pandas DataFrame of training (and validation) losses
    df = pd.DataFrame({'Frame': range(1, epochs + 1),
                       'MeanLoss': train_loss_list,
                       'LossType': 'Training'})
    if val_subj:
        tmp_df = pd.DataFrame({'Frame': range(1, epochs + 1),
                               'MeanLoss': val_loss_list,
                               'LossType': 'Validation'})
        df = pd.concat([df, tmp_df])
    
    # Save df
    df.to_csv(save_dir_path / 'mean_losses.csv', index=False)
    
    # Print update
    print(f'Losses saved here: "{save_dir_path / "mean_losses.csv"}"')

    # If required, estimate segmentations for validation dataset
    if val_subj:
        
        # Create a pandas DataFrame of Dice coefficients
        df = pd.DataFrame({'Frame': range(1, epochs + 1),
                        'MeanDSC': mean_dsc_list,
                        'Class': 'Overall'})
        
        # Convert mean_dsc_per_class from PyTorch tensor to NumPy array
        mean_dsc_per_class = mean_dsc_per_class.cpu().numpy()

        # Create a pandas DataFrame of Dice coefficients
        for idx, class_name in enumerate(['Head', 'SoftPalate', 'Jaw', 'Tongue', 'VocalTract', 'ToothSpace']):
            tmp_df = pd.DataFrame({'Frame': range(1, epochs + 1),
                                'MeanDSC': mean_dsc_per_class[idx, :],
                                'Class': class_name})
            df = pd.concat([df, tmp_df])
        
        # Save df
        df.to_csv(save_dir_path / 'mean_dsc.csv', index=False)

        # Print update
        print(f'Mean Dice coefficients saved here: "{save_dir_path / "mean_dsc.csv"}"')

        # Save segmentations estimated for validation dataset
        savemat(save_dir_path / 'estimated_segmentations.mat', {'est_segs': np.uint8(torch.permute(est_segs, (1, 2, 0)).cpu().numpy())})

        # Print update
        print(f'Segmentations estimated by segmentation CNN saved here: "{save_dir_path / "estimated_segmentations.mat"}"')


if __name__ == "__main__":
  
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Code to train a convolutional neural network to segment images.')
    
    # Add arguments
    parser.add_argument(
        '--data_dir', 
        help='The path to the folder containing all the data.',
        default='data',
        type=Path
        )
    parser.add_argument(
        '--train_subj',
        help='A list of ID(s) of subjects to use in the training dataset.',
        type=int,
        default=1,
        nargs='*'
        )
    parser.add_argument(
        '--val_subj',
        help='A list of ID(s) of subjects to use in the validation dataset.',
        type=int,
        default=2,
        nargs='*'
        )
    parser.add_argument(
        '--n_classes',
        help='Number of classes (including background) in ground-truth segmentations.',
        type=int,
        default=7
        )
    parser.add_argument(
        '--epochs',
        help='Number of epochs of training.',
        type=int,
        default=200
        )
    parser.add_argument(
        '--l_rate',
        help='Learning rate to use in training.',
        type=float,
        default=0.0003
        )
    parser.add_argument(
        '--mb_size',
        help='Number of images in a mini-batch.',
        type=int,
        default=4
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
    assert os.path.exists(args.data_dir), 'Please specify the absolute path to the folder containing all the data using the --data_dir argument to "CheckData.py".'

    # Check if images have been normalised
    assert os.path.exists(args.data_dir / 'Normalised_Images'), 'Have the images been normalised? This can be done using "NormaliseImages.py"'
    
    # If required, modify args.val_subj
    if args.val_subj == [-1]:
        args.val_subj = []

    # Run main function
    main(args.data_dir, args.train_subj, args.val_subj, args.n_classes, args.epochs, args.l_rate, args.mb_size)
