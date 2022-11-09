import os
import io
import zipfile
from pydicom.filereader import dcmread
import cv2
import numpy as np
from os.path import basename

directory = '/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive'
output_dir = '/dhc/home/tim.depping/GitHub/master_project/heart_images/'
counter = 0
for filename in os.listdir(directory)[:1]:
    f = os.path.join(directory, filename)
    if filename.endswith(".zip"):
        counter += 1
        archive = zipfile.ZipFile(f, 'r')
        first_file_name = archive.filelist[0]
        img_data = archive.read(first_file_name)
        ds = dcmread(io.BytesIO(img_data))
        img = ds.pixel_array
        # print("before conversion", img.tolist())
        img = img.astype(np.uint16)
        # print("after conversion", img.tolist())
        cv2.imwrite(f'{output_dir}{filename}_f1.png'.replace('.zip', ''),img) 
    if counter == 1000:
        break


zf = zipfile.ZipFile("images.zip", "w")
for filename in os.listdir(output_dir):
    print(filename)
    zf.write(os.path.join(output_dir, filename), '')
zf.close()