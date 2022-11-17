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
output_dir = '/dhc/home/tim.depping/GitHub/master_project/heart_images/'
output_file_name = "images"
counter = 0
max_counter = 1

debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if filename.endswith(".zip"):
        counter += 1
        archive = zipfile.ZipFile(f, 'r')
        ds = dcmread(io.BytesIO(archive.read(archive.filelist[0])))
        img = ds.pixel_array
        img = np.array(img, dtype = float) 
        img = (img - img.min()) / (img.max() - img.min()) * 255.0  
        img = img.astype(np.uint8)
        # new_img = np.zeros_like(img, dtype=np.uint16)
        # for iy, ix in np.ndindex(img.shape):
        #     new_img[iy][ix] = img[iy][ix]
        #     print(type(img[iy][ix]))
        png_filename = f'{output_dir}{filename}_f1.png'.replace('.zip', '')
        # test_img = np.arange(34944,dtype=np.uint16).reshape(168,208)
        debugpy.breakpoint()
        print(png_filename)
        imageio.imwrite(png_filename,img) 
        # cv2.imwrite(png_filename,test_img) 
    if counter == max_counter:
        break

zf = zipfile.ZipFile(f'{output_file_name}.zip', "w")
for filename in os.listdir(output_dir):
    zf.write(os.path.join(output_dir, filename), f'/{output_file_name}/{filename}')
zf.close()