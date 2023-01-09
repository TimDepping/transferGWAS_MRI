import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to input images (png)')
    parser.add_argument('--out_dir', type=str, default="/dhc/groups/mpws2022cl1/input/", help='Directory to save the csv file')
    parser.add_argument('--normalize_label', dest="normalize_label", action="store_true", default=False, help="normalize input labels")
    args = parser.parse_args()

    import_data = pd.read_csv("/dhc/groups/mpws2022cl1/input/cardio_44k.csv")

    # '22425-2.0': 'Cardiac Index'
    # '22420-2.0': 'Ejection Fraction'
    label = 'Ejection Fraction'
    label_code = '22420-2.0'

    df = import_data.loc[:, ['eid', label_code]].rename(columns={'eid': 'image', label_code: label})
    df = df.dropna(subset=[label])

    if (args.normalize_label):
        # Calculate the mean and standard deviation of the Cardiac index / Ejection Fraction column
        mean_value = df[label].mean()
        std_value = df[label].std()
        
        # Normalize the Cardiac index column
        df[label] = df[label].apply(lambda x: (x - mean_value) / std_value)
    
    df['image'] = df['image'].astype(str)
    df['image'] = df['image'].apply(lambda x: str(x) + '_CINE_segmented_LAX_4Ch_RGB_0-16-39')

    filenames = [name for name in os.listdir(args.img_dir) if os.path.splitext(name)[-1] == '.png']
    len(filenames)

    # Remove the ".png" ending from each filename in the list of filenames
    filenames = [filename[:-4] for filename in filenames]

    # Use the DataFrame.isin() method to check which rows in the DataFrame are in the list of filenames
    mask = df['image'].isin(filenames)

    # Use the boolean mask to filter the DataFrame and remove the rows that are not in the list of filenames
    df = df[mask]

    ## create filename string from label 
    filename_label = label.replace(" ", "")
    filename_label = filename_label[0].lower() + filename_label[1:]
    filename_label += "_"
    if (args.normalize_label):
        filename_label += "normalized_"

    available_labels = len(df)
    filename = filename_label + str(available_labels) + '.csv'
    filepath = os.path.join(args.out_dir, filename)

    df.to_csv(filepath, index=False)
    print('Created: ' + filepath + " successfully")


if __name__ == '__main__':
    main()
