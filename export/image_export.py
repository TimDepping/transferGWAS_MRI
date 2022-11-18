import os
import io
import zipfile
from pydicom.filereader import dcmread
import numpy as np
from os.path import basename
import imageio

directory = '/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive'
output_dir = './heart_image_export'
output_file_name = "heart_image_files"
counter = 0
max_counter = 100

os.makedirs(output_dir, exist_ok=True)
for patient_file_name in os.listdir(directory):
    if patient_file_name.endswith(".zip"):
        counter += 1
        archive = zipfile.ZipFile(os.path.join(directory, patient_file_name), 'r')
        for index, imaging_file in enumerate(archive.filelist):
            if imaging_file.filename.endswith(".dcm"):
                ds = dcmread(io.BytesIO(archive.read(imaging_file)))
                if ds.SeriesDescription == 'CINE_segmented_LAX_4Ch':
                    img = np.array(ds.pixel_array, dtype = float)
                    img = (img - img.min()) / (img.max() - img.min()) * 255.0  
                    img = img.astype(np.uint8)
                    png_filename = f'{output_dir}/{patient_file_name}_f1.png'.replace('.zip', '')
                    imageio.imwrite(png_filename,img)
                    break
    if counter == max_counter:
        break

zf = zipfile.ZipFile(f'{output_file_name}.zip', "w")
for filename in os.listdir(output_dir):
    zf.write(os.path.join(output_dir, filename), f'/{output_file_name}/{filename}')
zf.close()