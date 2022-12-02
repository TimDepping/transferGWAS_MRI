# This script exports the first 4Ch heart mri image of the participants. 

import os
import io
import csv
import zipfile
from pydicom.filereader import dcmread
import numpy as np
from os.path import basename
import imageio
import shutil

directory = '/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive'
output_file_name = "heart_image_files_TBD"
output_dir = f'.input/image_files/{output_file_name}'
counter = 0
max_counter = -1

if os.path.exists(output_dir) and os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

patients = []

for patient_file_name in os.listdir(directory):
    if patient_file_name.endswith(".zip"):
        archive = zipfile.ZipFile(os.path.join(directory, patient_file_name), 'r')
        for index, imaging_file in enumerate(archive.filelist):
            if imaging_file.filename.endswith(".dcm"):
                ds = dcmread(io.BytesIO(archive.read(imaging_file)))
                if ds.SeriesDescription == 'CINE_segmented_LAX_4Ch':
                    try:
                        img = np.array(ds.pixel_array, dtype = float)
                        img = (img - img.min()) / (img.max() - img.min()) * 255.0  
                        img = img.astype(np.uint8)
                        patient_id = patient_file_name.split("_")[0]
                        png_file_name = f'{patient_id}_f1.png'
                        png_file_path = f'{output_dir}/{png_file_name}'
                        patients.append({"IID": patient_id, "path": png_file_name})
                        imageio.imwrite(png_file_path,img)
                        counter += 1
                        print(f'Patient {counter} added.')
                    except:
                        print(f'Patient {counter} not added.')
                    break
    if counter == max_counter:
        break

with open(f'{output_file_name}.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["IID", "path"])
    writer.writeheader()
    writer.writerows(patients)

zf = zipfile.ZipFile(f'{output_file_name}.zip', "w")
for filename in os.listdir(output_dir):
    zf.write(os.path.join(output_dir, filename), f'/{output_file_name}/{filename}')
zf.close()