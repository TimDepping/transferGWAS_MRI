import SimpleITK as sitk
import os

input_dir = '1000346_20208_2_0'

series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_dir)
for id in series_IDs:
    print(id)


series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_dir, series_IDs[1])

for name in series_file_names:
    print(name)


# series_reader = sitk.ImageSeriesReader()
# series_reader.SetFileNames(series_file_names)
# series_reader.MetaDataDictionaryArrayUpdateOn()
# series_reader.LoadPrivateTagsOn()
# image3D = series_reader.Execute()


image_file_reader = sitk.ImageFileReader()
image_file_reader.SetImageIO("GDCMImageIO")
image_file_reader.SetFileName(series_file_names[0])
image_file_reader.ReadImageInformation()
image = image_file_reader.Execute()

print('Before modification:')
print('origin: ' + str(image.GetOrigin()))
print('size: ' + str(image.GetSize()))
print('spacing: ' + str(image.GetSpacing()))
print('direction: ' + str(image.GetDirection()))
print('pixel type: ' + str(image.GetPixelIDTypeAsString()))
print('number of pixel components: ' + str(image.GetNumberOfComponentsPerPixel()))

original_size = image.GetSize()
original_spacing = image.GetSpacing()
new_spacing = (1,1,1)
new_size = [int(original_size[0]*original_spacing[0]), int(original_size[1]*original_spacing[1]),original_size[2]]
image = sitk.Resample(
                image1=image,
                size=new_size,
                transform=sitk.Transform(),
                interpolator=sitk.sitkLinear,
                outputOrigin=image.GetOrigin(),
                outputSpacing=new_spacing,
                outputDirection=image.GetDirection(),
                defaultPixelValue=0,
                outputPixelType=image.GetPixelID(),
            )
image = sitk.RescaleIntensity(image, 0, 255)
image = sitk.Cast(image, sitk.sitkUInt8)
image = sitk.ScalarToRGBColormap(image)

writer = sitk.ImageFileWriter()
writer.SetFileName('processed.png')
writer.Execute(image)

print('After modification:')
print('origin: ' + str(image.GetOrigin()))
print('size: ' + str(image.GetSize()))
print('spacing: ' + str(image.GetSpacing()))
print('direction: ' + str(image.GetDirection()))
print('pixel type: ' + str(image.GetPixelIDTypeAsString()))
print('number of pixel components: ' + str(image.GetNumberOfComponentsPerPixel()))