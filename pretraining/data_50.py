from os.path import join

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

'''
3 data classes (MriData, AutoMriData, and RegressionMriData) representing datasets of MRI Data.
Subclasses of PyTorch's Dataset class.
The subset argument can be used to limit the number of items in the dataset (usefull for debugging).
'''

'''
Generic base class for AutoMriData and RegressionMriData for basic functionality.
Overwrites __len__ / __getitem__ to implement basic functionality of loading and preprocessing images and labels.
Returns img and label.
'''

class MriData(Dataset):
    # Constructor
    def __init__(
        self, path_col, label_col, tfms=None, subset=100, target_dtype=np.float32
    ):
        if subset:
            self.df = self.df.sample(subset, random_state=123)
        self.path_col = path_col
        self.label_col = label_col
        self.tfms = tfms
        self.target_dtype = target_dtype

    # Number of data samples in the dataset
    def __len__(self):
        return len(self.df)

    # loads a data sample, transforms it (for the quick fix)
    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
    
        path = self.df.iloc[idx][self.path_col]
        label = self.df.iloc[idx][self.label_col].astype(self.target_dtype)
        
        img = Image.open(path)
        return img, label

    # loads a item lable using load_item and transforms it 
    def __getitem__(self, idx):
        img, label = self._load_item(idx)
        if self.tfms:
            img = self.tfms(img)
        return img, label
    
    # def get_original_img(self, idx):
        # img, label = self._load_item(idx)
        # print("class label: %s, img shape: %s" % (label, img.size))
        # return img, label 
        



class MriData_50(Dataset):
    # Constructor
    def __init__(
        self, path_col, label_col, tfms=None, subset=100, target_dtype=np.float32
    ):
        if subset:
            self.df = self.df.sample(subset, random_state=123)
        self.path_col = path_col
        self.label_col = label_col
        self.tfms = tfms
        self.target_dtype = target_dtype

    # Number of data samples in the dataset
    def __len__(self):
        return len(self.df)

    # loads a data sample, transforms it (for the quick fix)
    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
    
        subject_path = self.df.iloc[idx][self.path_col]
        label = self.df.iloc[idx][self.label_col].astype(self.target_dtype)

        subject = os.path.basename(subject_path)
        
        images = []
        for i in range(50):
            file_name = os.path.join(subject + '_CINE_segmented_LAX_4Ch_' + str(i) + '.png')
            file_path = os.path.join(subject_path, file_name)
            image = Image.open(file_path).convert('L')
            images.append(np.array(image))

        image_data = np.array(images)
        image_data = np.transpose(image_data, (1, 2, 0))

        # convert numpy array to PIL image object (does not work)
        # img = Image.fromarray(image_data)
        
        return img, label

    # loads a item lable using load_item and transforms it 
    def __getitem__(self, idx):
        img, label = self._load_item(idx)
        if self.tfms:
            img = self.tfms(img)
        return img, label
    
    # def get_original_img(self, idx):
        # img, label = self._load_item(idx)
        # print("class label: %s, img shape: %s" % (label, img.size))
        # return img, label 


'''
Subclass of MriData specifically designed for use with autoencoders.
Returns only img as input and the target for each data sample (overrides methods of base class)
'''
class AutoMriData(MriData):
    def __init__(self, img_dir, labels_path, tfms=None, subset=100):
        ds = RegressionMriData(
            img_dir=img_dir, labels_path=labels_path, tfms=tfms, subset=subset
        )
        self.df = ds.df
        path_col = ds.path_col
        super().__init__(
            path_col=path_col,
            label_col=None,
            tfms=tfms,
            subset=None,
            target_dtype=np.float32,
        )

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.df.iloc[idx][self.path_col]
        img = Image.open(p)
        return img, None

    def __getitem__(self, idx):
        img, _ = self._load_item(idx)
        if self.tfms:
            img = self.tfms(img)
        return img, img

'''
Subclass of MriData designed for use with regression models.
Creates a df containing the paths to the images and their labels.
'''
class RegressionMriData(MriData):
    def __init__(
        self, img_dir, labels_path, tfms=None, subset=100, target_dtype=np.float32
    ):
        # df contains image names + label (e.g.1000096_f1,2.6)
        df = pd.read_csv(labels_path)
        df.image = [join(img_dir, p + ".png") for p in df.image]
        self.df = df[:subset]
        # self.df = df.drop([1146,])[:subset]
        super().__init__(
            path_col="image",
            # label_col="level",
            ##MRI
            label_col="Ejection Fraction",
            # label_col="Cardiac Index",
            tfms=tfms,
            subset=subset,
            target_dtype=target_dtype,
        )


class RegressionMriData_50(MriData_50):
    def __init__(
        self, img_dir, labels_path, tfms=None, subset=100, target_dtype=np.float32
    ):
        # df contains folder names + labels (e.g.1000096,2.6)
        df = pd.read_csv(labels_path)
        df.image = [join(img_dir, p) for p in df.image]
        self.df = df[:subset]
        # self.df = df.drop([1146,])[:subset]
        super().__init__(
            path_col="image",
            # label_col="level",
            ##MRI
            label_col="Ejection Fraction",
            # label_col="Cardiac Index",
            tfms=tfms,
            subset=subset,
            target_dtype=target_dtype,
        )

'''
Returns Transform Objects.
Train: Random Rotation, Resizing, ColorJitter, Random Horizontal Flip, ToTensor (Scaling), Normalization
Test: Resizing, ToTensor (Scaling), Normalization
'''
def get_tfms(size=224, interpolation=Image.BILINEAR):
    ## First 50000 images 
    mean = [0.1886, 0.1880, 0.1834]
    std = [0.1593, 0.1616, 0.1622]
    ## First 3000 images 
    # mean = [0.1880, 0.1876, 0.1829]
    # std = [0.1586, 0.1612, 0.1616]
    ## First 1000 images 
    # mean = [0.1874, 0.1870, 0.1825]
    # std = [0.1584, 0.1611, 0.1617]
    ## ImgNet mean and std
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)

    train = transforms.Compose(
        [
            transforms.RandomRotation(degrees=180, resample=interpolation),
            transforms.Resize(size=size),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.25, saturation=0.25, hue=0.03
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            norm,
        ]
    )
    valid = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(), norm,]
    )

    return train, valid

'''
Returns two DoataLoader Objects (training and validation)
'''
def build_mri_dataset(
    img_dir,
    labels_path,
    ae=False,
    size=224,
    batch_size=32,
    num_workers=12,
    # Split data into training and validation based on this parameter
    train_pct=0.8,
    subset=None,
    seed=123,
):
    # get transformation information
    train_tfm, valid_tfm = get_tfms(size=size, interpolation=Image.BILINEAR)

    # Create Dataset for Regression and AE
    if ae:
        train_ds = AutoMriData(
            img_dir=img_dir, labels_path=labels_path, tfms=train_tfm, subset=subset
        )
        valid_ds = AutoMriData(
            img_dir=img_dir, labels_path=labels_path, tfms=valid_tfm, subset=subset
        )
    else:
        train_ds = RegressionMriData(
            img_dir=img_dir, labels_path=labels_path, tfms=train_tfm, subset=subset
        )
        valid_ds = RegressionMriData(
            img_dir=img_dir, labels_path=labels_path, tfms=valid_tfm, subset=subset
        )

    # Split into training and validation set
    m = len(train_ds)
    cut = int(train_pct * m)
    torch.manual_seed(seed)
    inds = torch.randperm(m)
    train_inds = inds[:cut]
    valid_inds = inds[cut:]
    # shuffel images randomly
    train_sampler = SubsetRandomSampler(train_inds)
    valid_sampler = SubsetRandomSampler(valid_inds)

    # Create the dataLoaders for train and val
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,
    )

    # Returns two DataLoader objects (Training and Validation)
    return train_loader, valid_loader