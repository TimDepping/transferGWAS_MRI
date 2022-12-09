import os
import zipfile
from pydicom.filereader import dcmread
from os.path import basename
import debugpy
import csv
from io import TextIOWrapper
import shutil

directory = '/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive'
output_dir = './heart_dicom_export'
output_file_name = "heart_dicom_files"
counter = 0
max_counter = -1

if os.path.exists(output_dir) and os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
for patient_file_name in os.listdir(directory):
    if patient_file_name.endswith(".zip"):
        patient_id = patient_file_name.split("_")[0]
        with zipfile.ZipFile(os.path.join(directory, patient_file_name)) as patient_archive:
            CINE_segmented_LAX_4Ch_image_file_names = []
            
            manifest_file_name = 'manifest.csv' if 'manifest.csv' in patient_archive.namelist() else 'manifest.cvs'

            with patient_archive.open(manifest_file_name) as manifest_csv:
                manifest = csv.reader(TextIOWrapper(manifest_csv, 'utf-8'))
                CINE_segmented_LAX_4Ch_image_file_names = [row[0] for row in manifest if row[-3]=='CINE_segmented_LAX_4Ch']
            
            for index, file_name in enumerate(CINE_segmented_LAX_4Ch_image_file_names):
                # patient_output_dir = f'{output_dir}/{patient_id}'
                patient_archive.extract(file_name, path=output_dir)
                os.rename(f'{output_dir}/{file_name}', f'{output_dir}/{patient_id}_CINE_segmented_LAX_4Ch_{index}')

        counter += 1
    if counter == max_counter:
        break

shutil.make_archive(output_file_name, 'zip', output_dir)