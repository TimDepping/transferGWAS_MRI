import argparse
import os
from os.path import join
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

'''
This script creates the multichannel tensors needed for the training_main.py script.
One multichannel tensor stores all 50 greyscale images of one subject.
'''
def store_tensor(tensor, subject_id, out_dir):
    file_name = join(subject_id + '_CINE_segmented_LAX_4Ch_mc50.pt')
    file_path = join(out_dir, file_name)
    torch.save(tensor, file_path)

def create_tensors(img_dir, out_dir):
    ## Check if subject contains 50 png images. 
    subject_ids = tqdm([subject_id for subject_id in os.listdir(img_dir) if len(os.listdir(os.path.join(img_dir, subject_id))) >= 50])

    for subject_id in subject_ids:
        subject_path = os.path.join(img_dir, subject_id)
        tensors = []
        for i in range(50):
            file_name = join(subject_id + '_CINE_segmented_LAX_4Ch_' + str(i) + '.png')
            file_path = join(subject_path, file_name)
            img = Image.open(file_path)
            img_tensor = torch.from_numpy(np.array(img))
            img_tensor = torch.unsqueeze(img_tensor, dim=0) 
            tensors.append(img_tensor)

        stacked_tensor = torch.stack(tensors, dim=0)

        store_tensor(stacked_tensor, subject_id, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to multichannel input images')
    parser.add_argument('out_dir', type=str, help='Directory to save the multichannel tensors')
    args = parser.parse_args()

    # Create base tensors
    out_dir = args.out_dir
    os.makedirs(out_dir)
    create_tensors(args.img_dir, out_dir)

if __name__ == '__main__':
    main()
