import os
import io
import zipfile
from pydicom.filereader import dcmread
import cv2
import numpy as np
from os.path import basename
import debugpy
import imageio

directory = '/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive'
output_dir = './heart_dicom_export'
output_file_name = "heart_dicom_files"
counter = 0
max_counter = 100

os.makedirs(output_dir, exist_ok=True)
for patient_file_name in os.listdir(directory):
    if patient_file_name.endswith(".zip"):
        counter += 1
        archive = zipfile.ZipFile(os.path.join(directory, patient_file_name), 'r')
        dicom_files_4Ch = []
        for index, imaging_file in enumerate(archive.filelist):
            if imaging_file.filename.endswith(".dcm"):
                dicom_file_bytes = io.BytesIO(archive.read(imaging_file))
                ds = dcmread(io.BytesIO(archive.read(imaging_file)))
                if ds.SeriesDescription == 'CINE_segmented_LAX_4Ch':
                    dicom_files_4Ch.append(imaging_file)
        patient_file_name = patient_file_name.replace('.zip', '')
        dicom_filename = f'{output_dir}/{patient_file_name}.dcm'
        middle_dicom_file = dicom_files_4Ch[(len(dicom_files_4Ch)-1)//2]
        middle_dicom_file_bytes = io.BytesIO(archive.read(middle_dicom_file))
        with open(dicom_filename, "wb") as f:
            f.write(middle_dicom_file_bytes.getbuffer())
    if counter == max_counter:
        break

zf = zipfile.ZipFile(f'{output_file_name}.zip', "w")
for filename in os.listdir(output_dir):
    zf.write(os.path.join(output_dir, filename), f'/{output_file_name}/{filename}')
zf.close()