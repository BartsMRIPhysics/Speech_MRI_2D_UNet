# SpeechMRIDataset.py
# Code to create custom dataset for two-dimensional real-time 
# speech magnetic resonance (MR) images of the vocal tract during
# speech, and corresponding ground-truth (GT) segmentations

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 3rd March 2023

# Import required modules
from torch import is_tensor
from torch.utils.data import Dataset
from scipy.io import loadmat


# Create a custom dataset
class SpeechMRIDataset(Dataset):
    
    # Dataset of MR images and corresponding GT segmentations

    def __init__(self, img_path, seg_path, transform=None):
        
        # Args:
        # 1) img_path (string): Path to image file.
        # 2) seg_path (string): Path to corresponding GT segmentation file.
        # 3) transform (callable, optional): Optional transform to be applied on a sample.

        # Load csv file
        self.img_path = img_path
        self.seg_path = seg_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        # Load MAT-files into Python
        image_frame = loadmat(self.img_path[idx])
        mask_frame = loadmat(self.seg_path[idx])

        # Extract data from MAT-file
        image_frame = image_frame['image_frame']
        mask_frame = mask_frame['mask_frame']

        # Define sample
        sample = {'image_frame': image_frame, 'mask_frame': mask_frame}
        
        if self.transform:
            sample = self.transform(sample)

        return sample