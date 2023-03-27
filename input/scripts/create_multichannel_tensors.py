import argparse
import os
from os.path import join
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import csv
from pathlib import Path

"""
This script creates the multichannel tensors needed for the training_main.py script.
One multichannel tensor stores all 50 greyscale images of one subject.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str,
                        help='path to multichannel input images')
    parser.add_argument('out_dir', type=str,
                        help='Directory to save the multichannel tensors')
    args = parser.parse_args()

    img_dir = args.img_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    subject_ids = tqdm([
        subject_id
        for subject_id in os.listdir(img_dir)
        if len(os.listdir(os.path.join(img_dir, subject_id))) >= 50
    ])

    img_csv = []

    for subject_id in subject_ids:
        subject_path = os.path.join(img_dir, subject_id)
        tensors = []

        for i in range(50):
            file_name = f'{subject_id}_CINE_segmented_LAX_4Ch_{i}.png'
            file_path = join(subject_path, file_name)
            img = Image.open(file_path)
            img_tensor = torch.from_numpy(np.array(img))
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            tensors.append(img_tensor)

        # Save stacked tensor to file.
        output_file_name = f'{subject_id}_CINE_segmented_LAX_4Ch_mc50.pt'
        output_file_path = join(out_dir, output_file_name)
        stacked_tensor = torch.stack(tensors, dim=0)
        torch.save(stacked_tensor, output_file_path)

        # Prepare subject to be added to a csv file.
        img_csv.append({'IID': subject_id, 'path': output_file_name})

    # Write csv file with all subjects.
    output_path = Path(args.out_dir)
    with open(f'{output_path.absolute()}.csv', 'w') as csvfile:
        fieldnames = ['IID', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(img_csv)


if __name__ == '__main__':
    main()
