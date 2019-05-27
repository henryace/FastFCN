###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import numpy as np

import torch

from PIL import Image
from .base import BaseDataset

#
# Read & Modify 20190522
# Prepare custom dataset
# CMH
#

class CustomSegmentation(BaseDataset):
    BASE_DIR = 'CustomDataset'

    #
    # Original
    # NUM_CLASS = 150
    # CMH
    #

    NUM_CLASS = 2
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CustomSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using" + \
            "encoding/scripts/custom.py"

        #
        # function : _get_customdataset_pairs
        # read path : root
        # split : train / val / test
        # CMH
        #

        self.images, self.masks = _get_customdataset_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


class TpwDataset(BaseDataset):
    # BASE_DIR = 'CustomDataset'

    #
    # Original
    # NUM_CLASS = 150
    #
    # 20190524 add TpwDataset1
    # CMH
    #

    NUM_CLASS = 2
    def __init__(self, root=None, split='train',
                 mode=None, transform=None, img_folder = None, mask_folder = None,  target_transform=None, **kwargs):
        super(TpwDataset, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.images, self.masks = _get_customdataset_pairs(None, img_folder, mask_folder, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1



def _get_customdataset_pairs(folder = None , img_folder = None, mask_folder = None, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        if folder == None : 
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        #
        # Original
        # assert len(img_paths) == 20210
        # CMH
        #
    elif split == 'val':
        if folder == None : 
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        #
        # Original
        # assert len(img_paths) == 2000
        # CMH
        #
    elif split == 'test':
        folder = os.path.join(folder, '../release_test')
        with open(os.path.join(folder, 'list.txt')) as f:
            img_paths = [os.path.join(folder, 'testing', line.strip()) for line in f]
        #
        # Original
        # assert len(img_paths) == 3352
        # CMH
        #
        return img_paths, None
    else:
        assert split == 'trainval'
        #
        # train_img_folder : image folder for training
        # train_mask_folder : annotations folder for training
        # CMH
        #  
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        #
        # val_img_folder : image folder for validation
        # val_img_folder : annotations folder for validation
        # CMH
        #  
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        #
        # get_path_pairs : return img & mask pairs
        # train_img_paths, train_mask_paths
        # val_img_paths, val_mask_paths
        # CMH
        #  
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        # assert len(img_paths) == 22210
    return img_paths, mask_paths
