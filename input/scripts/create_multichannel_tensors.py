import argparse
import os
from os.path import join
import torch
from torchvision import transforms
from PIL import Image

# 50k Grey
MEAN=[0.1894]
STD=[0.1609]

norm = transforms.Normalize(mean=MEAN, std=STD)

TRAIN = transforms.Compose(
    [
        transforms.RandomRotation(degrees=180, resample=Image.BILINEAR),
        transforms.Resize(size=224),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.25, saturation=0.25, hue=0.03
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        norm,
    ]
)
VALID = transforms.Compose(
    [transforms.Resize(size=224), transforms.ToTensor(), norm,]
)

def store_tensor(tensor, subject_id, out_dir):
    file_name = join(subject_id + '_CINE_segmented_LAX_4Ch_mc50.pt')
    file_path = join(out_dir, file_name)
    torch.save(tensor, file_path)

def create_tensors(img_dir, out_dir, tfms):

    ## Check if subject contains 50 png images. 
    subject_ids = [subject_id for subject_id in os.listdir(img_dir) if len(os.listdir(os.path.join(img_dir, subject_id))) >= 50]

    for subject_id in subject_ids:
        subject_path = os.path.join(img_dir, subject_id)
        tensors = []
        for i in range(50):
            file_name = join(subject_id + '_CINE_segmented_LAX_4Ch_' + str(i) + '.png')
            file_path = join(subject_path, file_name)
            img = Image.open(file_path)

            if tfms:
                img_tensor = tfms(img)
            tensors.append(img_tensor)

        stacked_tensor = torch.stack(tensors, dim=2)
        stacked_tensor = stacked_tensor.reshape([50,224,224])

        store_tensor(stacked_tensor, subject_id, out_dir)

def create_log_file(out_dir, tfms):
    path_name = join(out_dir, "log.txt")
    with open(path_name, "w") as file:
        # write variables to the file
        file.write("tfms: " + str(tfms) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to multichannel input images')
    parser.add_argument('out_dir', type=str, help='Directory to save the multichannel tensors')
    args = parser.parse_args()

    # Create train tensors
    tfms = TRAIN
    out_dir = join(args.out_dir, "train")
    os.makedirs(out_dir)
    create_tensors(args.img_dir, out_dir, tfms)
    create_log_file(out_dir, tfms)

    # Create valid tensors
    tfms = VALID
    out_dir = join(args.out_dir, "valid")
    os.makedirs(out_dir)
    create_tensors(args.img_dir, out_dir, tfms)
    create_log_file(out_dir, tfms)

if __name__ == '__main__':
    main()
