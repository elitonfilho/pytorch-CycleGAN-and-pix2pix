"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose
import albumentations.augmentations.transforms as tr
from PIL import Image
import numpy as np
import cv2
from skimage import io


class V2Dataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=99999, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        # get the image paths of your dataset;
        self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transformv2(opt)
        self.final_transforms = get_final_transforms_v2(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = np.array(io.imread(A_path))
        B_img = np.array(cv2.imread(B_path))

        transformed = self.transform(image=A_img, imageB=B_img)

        _data_A = transformed['image']
        _data_B = transformed['imageB']

        final_transforms_A, final_transforms_B = self.final_transforms
        data_A = final_transforms_A(image=_data_A)['image']
        data_B = final_transforms_B(image=_data_B)['image']

        # return {'data_A': data_A, 'data_B': data_B, 'path': path}
        return {'A': data_A, 'B': data_B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)


def get_transformv2(opt):
    transform_list = []
    # Transforms in opt.preprocess
    if 'fixsize' in opt.preprocess:
        transform_list.append(tr.Resize(286, 286, interpolation=2, p=1))
    if 'resize' in opt.preprocess:
        transform_list.append(tr.Resize(opt.load_size, opt.load_size, interpolation=2, p=1))
    if 'crop' in opt.preprocess:
        transform_list.append(tr.RandomCrop(opt.crop_size, opt.crop_size, p=1))
    # Transforms in colorspace
    if 'color' in opt.preprocess:
        transform_list.extend([
            tr.RandomContrast(limit=0.2, p=0.5),
            tr.RandomBrightness(limit=0.2, p=0.5),
            tr.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # tr.ISONoise()
        ])
    # Necessary transforms
    transform_list.extend([
        tr.HorizontalFlip(p=0.5),
        tr.VerticalFlip(p=0.5)
    ])
    return Compose(transform_list, additional_targets={'imageB':'image'})

def get_final_transforms_v2(opt):
    compose_A = Compose([
        tr.ToFloat(max_value=1024),
        ToTensorV2()
    ])
    compose_B = Compose([
        tr.ToFloat(max_value=256),
        ToTensorV2()
    ])
    return compose_A, compose_B
