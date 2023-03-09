# CheckData.py
# Code to check that the speech magnetic resonance images 
# and corresponding ground-truth (GT) segmentations are 
# have not been corrupted and are organised correctly

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 3rd March 2023

# Import required modules
import argparse
from pathlib import Path
import os
from pydicom import dcmread
from scipy.io import loadmat
from numpy import min, max, unique

def main(data_dir, subj_id_list):

    # For each subject ID
    for subj_id in subj_id_list:

        # Check that the Image and GT_Segmentations folders contain subfolders with this subject ID
        assert os.path.exists(data_dir / 'MRI_SSFP_10fps' / f'Subject{subj_id}'), f'The "SSFP_MRI_10fps" folder does not contain a folder called "Subject{subj_id}"'
        assert os.path.exists(data_dir / 'GT_Segmentations' / f'Subject{subj_id}'), f'The "GT_Segmentations" folder does not contain a folder called "Subject{subj_id}"'

    # Preallocate list for paths to images and corresponding 
    # ground-truth (GT) segmentations to include in dataset
    full_img_list = []
    full_seg_list = []

    # For each subject ID
    for subj_id in subj_id_list:

        # Create lists of frames in the subfolders
        img_list = [int(g[6:-4]) for g in os.listdir(data_dir / 'MRI_SSFP_10fps' / f'Subject{subj_id}') if (g.endswith('.dcm') and g.startswith('image'))]
        img_list.sort()
        seg_list = [int(g[5:-4]) for g in os.listdir(data_dir / 'GT_Segmentations' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('mask'))]
        seg_list.sort()

        # Check that subfolders contain frames
        assert img_list, f'There are no files named image_N.mat in folder {data_dir / "MRI_SSFP_10fps" / f"Subject{subj_id}"}'
        assert seg_list, f'There are no files named mask_N.mat in folder {data_dir / "GT_Segmentations" / f"Subject{subj_id}"}'

        # Check that corresponding frames are contained in the subfolders
        assert img_list == seg_list, f'There is a discrepancy in the IDs of the images in folder {data_dir / "MRI_SSFP_10fps" / f"Subject{subj_id}"} and the ground-truth segmentations in folder {data_dir / "GT_Segmentations" / f"Subject{subj_id}"}'

        # Print update
        print(f'Subject {subj_id} has {len(img_list)} images and corresponding ground-truth segmentations')

        # Create lists of paths to images and GT segmentations
        img_list = [data_dir / 'MRI_SSFP_10fps' / f'Subject{subj_id}' / f'image_{g}.dcm' for g in img_list]
        seg_list = [data_dir / 'GT_Segmentations' / f'Subject{subj_id}' / f'mask_{g}.mat' for g in seg_list]

        # Update full_img_list and full_seg_list
        full_img_list += img_list
        full_seg_list += seg_list

    # Load first image
    frame = dcmread(full_img_list[0]).pixel_array

    # Image dimensions
    first_frame_dim = frame.shape

    # Check image is two-dimensional
    assert len(first_frame_dim) == 2, f'{full_img_list[0]} is not two-dimensional'

    # For each image
    for file_path in full_img_list:

        # Load image
        frame = dcmread(file_path).pixel_array

        # Image dimensions
        frame_dim = frame.shape

        # Check image dimensions are consistent
        assert frame_dim == first_frame_dim, f'Not all images in folder "{file_path.parent}" have the same dimensions'

        # Check that image pixel values are all zero or greater
        assert min(frame) >= 0, f'The pixel values of images in folder "{file_path.parent}" are less than 0'

    # Print update
    print('Images are organised correctly and are not corrupted')

    # For each segmentation
    for file_path in full_seg_list:

        # Load segmentation
        frame = loadmat(file_path)['mask_frame'].astype('int64')

        # Segmentation dimensions
        frame_dim = frame.shape

        # Check segmentation dimensions are consistent
        assert frame_dim == first_frame_dim, f'Not all segmentations in folder {file_path.parent} have the same dimensions'

        # Check segmentation consists of seven classes (including background)
        seg_classes = unique(frame)
        assert len(seg_classes) == 7, f'Not all segmentations in folder {file_path.parent} have seven classes (including the background)'
        assert min(seg_classes) == 0, f'Not all segmentations in folder {file_path.parent} have a minimum pixel value of 0'
        assert max(seg_classes) == 6, f'Not all segmentations in folder {file_path.parent} have a maximum pixel value of 6'

    # Print update
    print('Ground-truth segmentations are organised correctly and are not corrupted')

    # Print update
    print('Next step: run "NormaliseImages.py" to normalise the images')

if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description='Code to check that the images and ground-truth segmentations are organised correctly and are not corrupted.')
    
    # Add arguments
    parser.add_argument(
        '--data_dir', 
        help='Path to the folder containing all the data.',
        default='data',
        type=Path
        )
    parser.add_argument(
        '--subj_id_list',
        help='List of IDs of subjects whose data should be checked.',
        default=[1, 2, 3, 4, 5],
        type=list
        )
    
    # Parse arguments
    args = parser.parse_args()

    # Check if data_dir exists
    assert os.path.exists(args.data_dir), 'Please specify the absolute path to the folder containing all the data using the --data_dir argument to "CheckData.py".'

    # Run main function
    main(args.data_dir, args.subj_id_list)