import argparse
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class MriData(Dataset):
    def __init__(self, img_dir, tfms=None):
        self.tfms = tfms
        self.data = []
        for patient_dir in os.listdir(img_dir):
            patient_path = os.path.join(img_dir, patient_dir)
            filenames = [name for name in os.listdir(patient_path) if os.path.splitext(name)[-1] == '.png']
            self.data.extend([(os.path.join(patient_path, p)) for p in filenames])

    def __len__(self):
        return len(self.data)

    def _load_item(self, idx):    
        path = self.data[idx]
        img = Image.open(path).convert('L')
        return img, 0

    def __getitem__(self, idx):
        img, _ = self._load_item(idx)

        if self.tfms:
            img = self.tfms(img)
        else:
            transform = transforms.Compose(
                [
                transforms.ToTensor(),            
                ]
            )
            img = transform(img)
        return img, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to input images (png)')
    args = parser.parse_args()

    image_data = MriData(args.img_dir) 

    loader = DataLoader(
        image_data, 
        batch_size = 20, 
        num_workers=1)

    mean, std = batch_mean_and_sd(loader)
    print("mean and std: \n", mean, std)  


def batch_mean_and_sd(loader):
    # Initialize count, first moment and second moment
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    # Iterate over the batches of images from the loader
    for images, _ in loader:
        # Unpack the shape of the images
        b, c, h, w = images.shape
        # Calculate the number of pixels
        nb_pixels = b * h * w
        # Calculate the sum of the images
        sum_ = torch.sum(images, dim=[0, 2, 3])
        # Calculate the sum of squares of the images
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        # Update the running sum
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        # Update the running sum of squares
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        # Increase the pixel count
        cnt += nb_pixels

    # Calculate the mean and standard deviation
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    # Return the mean and standard deviation
    return mean, std
        
if __name__ == '__main__':
    main()
