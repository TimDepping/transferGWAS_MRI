import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to input images (png)')
    parser.add_argument('--out_dir', type=str, default="/dhc/groups/mpws2022cl1/input/", help='Directory to save the csv file')
    args = parser.parse_args()

    import_data = pd.read_csv("/dhc/groups/mpws2022cl1/input/cardio_44k.csv")

    df = import_data.loc[:, ['eid', '22420-2.0']].rename(columns={'eid': 'image', '22420-2.0': 'Ejection Fraction'})
    df = df.dropna(subset=['Ejection Fraction'])

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

    available_labels = len(df)
    filename = 'ejectionFraction_' + str(available_labels) + '.csv'
    filepath = os.path.join(args.out_dir, filename)

    df.to_csv(filepath, index=False)
    print('Created: ' + filepath + " successfully")


if __name__ == '__main__':
    main()
