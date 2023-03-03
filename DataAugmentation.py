# DataAugmentation.py
# Code to perform data augmentation

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 3rd March 2023

# Import required modules
from numpy import float32, int64, newaxis, float64, zeros_like
from torch import from_numpy
from numpy.random import uniform, randint
from skimage.transform import rotate, resize
from math import floor, ceil, sin, radians


# Code to convert images and corresponding ground-truth 
# (GT) segmentations from NumPy arrays to PyTorch tensors

class ToTensor(object):
    
    # To convert NumPy arrays in a sample to PyTorch tensors

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        # Add channel to image_frame and convert to correct data type
        image_frame = float32(image_frame[newaxis, :, :])

        # Convert mask_frame to correct data type
        mask_frame = int64(mask_frame)

        return {'image_frame': from_numpy(image_frame),
                'mask_frame': from_numpy(mask_frame)}


# Code to rotate, crop and then zero pad images and 
# corresponding GT segmentations
class RotateCropAndPad(object):
    
    # Rotate, crop and then zero pad the image and 
    # GT segmentations in a sample so that their 
    # dimensions do not change
    # Arg:
    # 1) rotation_range (tuple) the range of angles 
    #    in degrees that the image can be rotated by

    def __init__(self, rotation_range):
        self.rotation_range = rotation_range

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        # Find size of image and mask
        h, w = image_frame.shape

        # Randomly choose rotation
        rotation_angle = uniform(self.rotation_range[0], self.rotation_range[1])

        # Convert mask_frame to correct data type
        mask_frame = float64(mask_frame)

        # Rotate image_frame and mask_frame
        image_frame = rotate(image_frame, rotation_angle, order = 0) # 0: nearest neighbour
        mask_frame = rotate(mask_frame, rotation_angle, order = 0) # 0: nearest neighbour

        # Define rotated, cropped and padded image frame and mask frame
        rotated_image_frame = zeros_like(image_frame)
        rotated_mask_frame = zeros_like(mask_frame)

        # Convert rotation_angle to radians
        rotation_angle = radians(rotation_angle)

        # Crop and pad rotated image frame and mask frame
        if rotation_angle > 0:
            rotated_image_frame[ceil(w / 2 * sin(rotation_angle)):, :] = image_frame[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_mask_frame[ceil(w / 2 * sin(rotation_angle)):, :] = mask_frame[:floor(h - w / 2 * sin(rotation_angle)), :]
        else:
            rotated_image_frame[ceil(w / 4 * -sin(rotation_angle)):, :] = image_frame[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_mask_frame[ceil(w / 4 * -sin(rotation_angle)):, :] = mask_frame[:floor(h + w / 4 * sin(rotation_angle)), :]

        return {'image_frame': rotated_image_frame, 'mask_frame': rotated_mask_frame}
    

# Code to crop images and corresponding GT segmentations
class RandomCrop(object):
    
    # Random crop with top left corner within x_range and y_range
    # Args:
    # 1) x_range (tuple) the range of possible x-coordinates for 
    #    the top left corner of the crop
    # 2) y_range (tuple) the range of possible y-coordinates for 
    #    the top left corner of the crop
    # 3) output_size (int or tuple) the dimensions of the crop. If 
    #    int, the crop is square

    def __init__(self, x_range, y_range, output_size):
        self.x_range = x_range
        self.y_range = y_range
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        new_h, new_w = self.output_size

        top = randint(self.y_range[0], self.y_range[1])
        left = randint(self.x_range[0], self.x_range[1])

        image_frame = image_frame[top: top + new_h,
                      left: left + new_w]

        mask_frame = mask_frame[top: top + new_h,
                      left: left + new_w]

        return {'image_frame': image_frame, 'mask_frame': mask_frame}
    

# Code to rescale images and corresponding GT segmentations
class Rescale(object):
    
    # Rescale the image and mask in a sample to a given size
    # Arg:
    # 1) output_size (int or tuple) the desired output size. If 
    #    int, the size of the smaller edge is matched to output_size 
    #    while keeping the aspect ratio the same

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        h, w = image_frame.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        # Convert mask_frame to correct data type
        mask_frame = float64(mask_frame)

        # Resize image_frame and mask_frame
        image_frame = resize(image_frame, (new_h, new_w), order = 0) # 0: nearest neighbour
        mask_frame = resize(mask_frame, (new_h, new_w), order = 0) # 0: nearest neighbour)

        return {'image_frame': image_frame, 'mask_frame': mask_frame}


# Code to translate images and corresponding GT segmentations
class RandomTranslation(object):
    
    # Random translation with top left corner within x_range and y_range
    # Args:
    # 1) x_range (tuple) the range of possible x-coordinates 
    #    for the top left corner of the translated image
    # 2) y_range (tuple) the range of possible y-coordinates 
    #    for the top left corner of the translated image

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        # Find size of image and mask
        h, w = image_frame.shape

        # Randomly choose translation
        top = randint(self.y_range[0], self.y_range[1])
        left = randint(self.x_range[0], self.x_range[1])

        # Define translated image_frame and mask_frame
        translated_image_frame = zeros_like(image_frame)
        translated_mask_frame = zeros_like(mask_frame)

        # Populate translated_image_frame and translated_mask_frame
        if left > 0:
            translated_image_frame[top:, left:] = image_frame[:h - top, :w - left]
            translated_mask_frame[top:, left:] = mask_frame[:h - top, :w - left]
        else:
            translated_image_frame[top:, :w + left] = image_frame[:h - top:, -left:]
            translated_mask_frame[top:, :w + left] = mask_frame[:h - top, -left:]

        return {'image_frame': translated_image_frame, 'mask_frame': translated_mask_frame}
    
# Code to rescale and the zero pad images and 
# corresponding GT segmentations
class RescaleAndPad(object):
    
    # Rescale and then zero pad the image and GT segmentations in a sample 
    # so that their dimensions do not change
    # Arg:
    # 1) resize_range (tuple) the range of acceptable matrix dimensions 
    #    following the rescale

    def __init__(self, resize_range):
        self.resize_range = resize_range

    def __call__(self, sample):
        image_frame, mask_frame = sample['image_frame'], sample['mask_frame']

        # Find size of image and mask
        h, w = image_frame.shape

        # Randomly choose new size of image from within resize_range
        new_size = randint(self.resize_range[0], self.resize_range[1])

        # Randomly choose lateral translation
        dx = randint(0, w - new_size)

        # Convert mask_frame to correct data type
        mask_frame = float64(mask_frame)

        # Preallocate arrays for rescaled and zero padded image and mask
        new_image_frame = zeros_like(image_frame)
        new_mask_frame = zeros_like(mask_frame)

        # Resize image_frame and mask_frame
        image_frame = resize(image_frame, (new_size, new_size), order = 0) # 0: nearest neighbour
        mask_frame = resize(mask_frame, (new_size, new_size), order = 0) # 0: nearest neighbour)

        # Zero pad resized image_frame and mask_frame
        new_image_frame[h - new_size:, dx:dx + new_size] = image_frame
        new_mask_frame[h - new_size:, dx:dx + new_size] = mask_frame

        return {'image_frame': new_image_frame, 'mask_frame': new_mask_frame}