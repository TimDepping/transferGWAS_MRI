import os
import zipfile
from os.path import basename
import shutil
import SimpleITK as sitk
import argparse
import csv
from pathlib import Path

# output_dir = f'/dhc/groups/mpws2022cl1/input/image_files/heart_image_files/heart_image_files_dicom/heart_image_files_10'

def resample_spacing(image):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = (1,1)
    new_size = [int(original_size[0]*original_spacing[0]), int(original_size[1]*original_spacing[1])]
    return sitk.Resample(
                    image1=image,
                    size=new_size,
                    transform=sitk.Transform(),
                    interpolator=sitk.sitkLinear,
                    outputOrigin=image.GetOrigin(),
                    outputSpacing=new_spacing,
                    outputDirection=image.GetDirection(),
                    defaultPixelValue=0,
                    outputPixelType=image.GetPixelID())
    
        
def crop_to_square(image):
    extractor = sitk.ExtractImageFilter()
    width, height = image.GetSize()
    square_size = min(width, height, 224)
    extractor.SetSize([square_size, square_size])
    middle_index = [width / 2, height / 2]
    half_square_size = int(square_size / 2)
    start_index = tuple(map(int, [middle_index[0] - half_square_size, middle_index[1] - half_square_size]))
    extractor.SetIndex(start_index)
    image = extractor.Execute(image)
    return image

def preprocess_dicom_to_png(image):
    image = resample_spacing(image)
    
    # Convert image to 8-bit RGB.
    image = sitk.RescaleIntensity(image, 0, 255)
    image = sitk.Cast(image, sitk.sitkUInt8)
    image = sitk.ScalarToRGBColormap(image)

    image = crop_to_square(image)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/dhc/projects/ukbiobank/original/imaging/heart_mri/lax/archive', help='Path to input directory.')
    parser.add_argument('--temp_dir', type=str, default='./.tmp', help='Path to temporary directory.')
    parser.add_argument('--output_dir', type=str, default='./out/10_RGB_0-16-39', help='Path to output directory. By default it is ./out/10_RGB_0-16-39, because the images with the indices 0,16,39 of the first 10 patients are used to export 10 RGB images which combines the three images for each patient.')
    parser.add_argument('--create_subdirs', type=bool, default=False, help='Create a subdirectory for every patient.')
    parser.add_argument('--export_format', type=str, default='png', choices=['png, dcm'], help='Export format for image files.')
    parser.add_argument('--create_zip', type=bool, default=False, help='Create a zip file of the output directory.')
    parser.add_argument('--number_of_patients', type=int, default=10, help='Number of patients to get image files from. By default images of 10 patients are exported. Set to -1 to export images of all patients.')
    parser.add_argument('--indices', type=str, default="0,16,39", help='Indices of images to export e.g. "0,16,39". By default all images are exported.')
    parser.add_argument('--export_RGB_from_indices', type=str, default=True, help='Requires that exactly three indices are provided. Export one RGB image that combines the three greyscale images. The create_subdirs argument will be ignored.')
    args = parser.parse_args()

    indices = []
    if len(args.indices) >= 1:
        indices = [int(item) for item in args.indices.split(',')]

    # Prepare directories.
    dirs = [args.output_dir, args.temp_dir]
    for dir in dirs:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    
    img_csv = []
    counter = 0
    for patient_file_name in os.listdir(args.input_dir):
        if patient_file_name.endswith(".zip"):
            patient_id = patient_file_name.split("_")[0]
            # print(f'patient: {patient_id}')
            with zipfile.ZipFile(os.path.join(args.input_dir, patient_file_name),'r') as patient_archive:
                patient_temp_dir = f'{args.temp_dir}/{patient_id}'
                patient_archive.extractall(f'{patient_temp_dir}')
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(patient_temp_dir)
                for series_id in series_ids:
                    series_reader = sitk.ImageSeriesReader()
                    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(patient_temp_dir, series_id)
                    series_reader.SetFileNames(series_file_names)
                    series_reader.MetaDataDictionaryArrayUpdateOn()
                    # series_reader.LoadPrivateTagsOn()
                    images = series_reader.Execute()
                    series_description_key = '0008|103e'
                    series_description = series_reader.GetMetaData(0, series_description_key)

                    if series_description == 'CINE_segmented_LAX_4Ch':
                        if args.export_RGB_from_indices:
                            image_1 = images[:, :, indices[0]]
                            image_2 = images[:, :, indices[1]]
                            image_3 = images[:, :, indices[2]]
                            image_2_resampled = sitk.Resample(image_2, image_1)
                            image_3_resampled = sitk.Resample(image_3, image_1)
                            image = sitk.Compose([image_1, image_2_resampled, image_3_resampled])
                            image = resample_spacing(image)

                            image = sitk.RescaleIntensity(image, 0, 255)
                            image = sitk.Cast(image, sitk.sitkVectorUInt8)
                            image = crop_to_square(image)
                            
                            # print('origin: ' + str(image.GetOrigin()))
                            # print('size: ' + str(image.GetSize()))
                            # print('spacing: ' + str(image.GetSpacing()))
                            # print('direction: ' + str(image.GetDirection()))
                            # print('pixel type: ' + str(image.GetPixelIDTypeAsString()))
                            # print('number of pixel components: ' + str(image.GetNumberOfComponentsPerPixel()))

                            indices_text = '-'.join(map(str,indices))
                            file_name = f'{patient_id}_{series_description}_RGB_{indices_text}.{args.export_format}'
                            writer = sitk.ImageFileWriter()
                            writer.SetFileName(f'{args.output_dir}/{file_name}')
                            writer.Execute(image)
                            img_csv.append({'IID': patient_id, 'path': file_name})
                        else:
                            for i in range(images.GetDepth()):
                                if len(indices) != 0 and i not in indices:
                                    continue
                                image = images[:, :, i]

                                # Preprare output directory.
                                patient_output_dir = args.output_dir
                                if args.create_subdirs:
                                    patient_output_dir = f'{args.output_dir}/{patient_id}'
                                    os.makedirs(patient_output_dir, exist_ok=True)
                                file_name = f'{patient_id}_{series_description}_{i}.{args.export_format}'
                                file_output_path = f'{patient_output_dir}/{file_name}'

                                if args.export_format == 'png':
                                    image = preprocess_dicom_to_png(image)
                                
                                writer = sitk.ImageFileWriter()
                                writer.SetFileName(file_output_path)
                                writer.Execute(image)
                                img_csv.append({'IID': patient_id,
                                                'path': f'{patient_id}/{file_name}' if args.create_subdirs else file_name,
                                                'instance': i})
            counter += 1
        if counter == args.number_of_patients:
            break
    shutil.rmtree(args.temp_dir)
    
    output_path = Path(args.output_dir)
    with open(f'{output_path.absolute()}.csv', 'w') as csvfile:
        fieldnames = ['IID', 'path', 'instance']
        if args.export_RGB_from_indices:
            del fieldnames[-1]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(img_csv)

    if args.create_zip:
        shutil.make_archive(f'{output_path.absolute()}', 'zip', output_path)

if __name__ == "__main__":
    main()