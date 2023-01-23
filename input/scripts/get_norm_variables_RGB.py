import argparse

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

""" DataClass to feed DataLoader """
class MriData(Dataset):
    # Constructor
    def __init__(
        self, img_dir, tfms=None
    ):
        self.tfms = tfms
        filenames = [name for name in os.listdir(img_dir) if os.path.splitext(name)[-1] == '.png']
        # Initialize the dataframe with a list of file paths
        df = pd.DataFrame({'image': [os.path.join(img_dir, p) for p in filenames]})
        self.df = df
        
    # Number of data samples in the dataset
    def __len__(self):
        return len(self.df)

    # loads a data sample
    def _load_item(self, idx):    
        path = self.df.iloc[idx]["image"]
        img = Image.open(path)
        return img, 0

    # loads a item lable using load_item 
    def __getitem__(self, idx):
        img, _ = self._load_item(idx)

        transform = transforms.Compose(
            [
            transforms.ToTensor(),            
            ]
        )
        
        img = transform(img)

        return img, 0

""" 
This script returns the mean and std of a specified dataset (input images). 
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to input images (png)')
    parser.add_argument('batch_size', type=int, help='batch size')
    args = parser.parse_args()

    ds = MriData(args.img_dir)
    m = len(ds)
    print("Number of Inputs: ", m)
    image_data = MriData(args.img_dir)

    loader = DataLoader(
        image_data, 
        batch_size = args.batch_size, 
        num_workers=1)

    mean, std = batch_mean_and_sd(loader)
    print("mean and std: \n", mean, std)  

    
def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std


if __name__ == '__main__':
    main()
