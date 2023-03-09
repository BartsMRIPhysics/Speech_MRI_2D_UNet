# NormaliseImages.py
# Code to normalise the speech magnetic resonance images

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 9th March 2023

# Import required modules
import argparse
from pathlib import Path
import os
from pydicom import dcmread
from numpy import zeros, min, max, single
from scipy.io import savemat

def main(data_dir, subj_id_list):

    # For each subject ID
    for subj_id in subj_id_list:

        # Print update
        print(f'Normalising images of subject {subj_id}')

        # Create list of images in the Subject<ID> subfolder
        img_list = [int(g[6:-4]) for g in os.listdir(data_dir / 'MRI_SSFP_10fps' / f'Subject{subj_id}') if (g.endswith('.dcm') and g.startswith('image'))]
        img_list.sort()
        img_list = [data_dir / 'MRI_SSFP_10fps'/ f'Subject{subj_id}' / f'image_{g}.dcm' for g in img_list]

        # Load first image
        img = dcmread(img_list[0]).pixel_array

        # Image dimensions
        img_dim = img.shape

        # Preallocate NumPy array for images
        all_img = zeros((img_dim[0], img_dim[1], len(img_list)))

        # For each image
        for idx, img_name in enumerate(img_list):

            # Read image
            img = dcmread(img_name).pixel_array

            # Populate all_img
            all_img[..., idx] = img

        # Normalise image series
        all_img = (all_img - min(all_img)) / (max(all_img) - min(all_img))

        # Check normalisation
        assert min(all_img) >= 0, f'Subject {subj_id} images have not been normalised'
        assert max(all_img) <= 1, f'Subject {subj_id} images have not been normalised'

        # Path to folder where normalised images will be saved
        save_dir_path = data_dir / 'Normalised_Images' / f'Subject{subj_id}'

        # If required, create folder where normalised images will be saved
        if os.path.exists(save_dir_path):

            # Print update
            print(f'"{save_dir_path}" already exists')
        
        else:

            # Create folders
            os.makedirs(save_dir_path)

            # Print update
            print(f'"{save_dir_path}" created')

        # Print update
        print('Starting to save normalised images')

        # For each normalised image in the series
        for idx in range(0, len(img_list)):

            # Save normalised image as a MAT file
            savemat(save_dir_path / f'image_{idx + 1}.mat', {'image_frame': all_img[..., idx]})

        # Print update
        print('Finished saving normalised images')

    # Print update
    print('Finished normalising and saving all images')

if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description='Code to normalise images.')
    
    # Add arguments
    parser.add_argument(
        '--data_dir', 
        help='Path to the folder containing all the data.',
        default='data',
        type=Path
        )
    parser.add_argument(
        '--subj_id_list',
        help='List of IDs of subjects whose images should be normalised.',
        default=[1, 2, 3, 4, 5],
        type=list
        )
    
    # Parse arguments
    args = parser.parse_args()

    # Check if data_dir exists
    assert os.path.exists(args.data_dir), 'Please specify the absolute path to the folder containing all the data using the --data_dir argument to "NormaliseImages.py".'

    # Run main function
    main(args.data_dir, args.subj_id_list)