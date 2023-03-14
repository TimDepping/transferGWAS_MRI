import os
from os.path import basename
import csv
from pathlib import Path


def main():
    input_dir = "/dhc/groups/mpws2022cl1/tensor/50000_GRAY"
    output_file = '/dhc/groups/mpws2022cl1/tensor/50000_GRAY.csv'
    img_csv = []
    for patient_file_name in os.listdir(input_dir):
        patient_id = patient_file_name.split("_")[0]
        img_csv.append({'IID': patient_id, 'path': patient_file_name})

    output_path = Path(output_file)
    with open(f'{output_path.absolute()}', 'w') as csvfile:
        fieldnames = ['IID', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(img_csv)


if __name__ == "__main__":
    main()
