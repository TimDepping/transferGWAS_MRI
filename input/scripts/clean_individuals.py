import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indiv_csv', type=str, help='path to csv file holding IID & path to img')
    parser.add_argument('exclude_csv', type=str, help='path to csv file holding IIDs to exclude')
    parser.add_argument('--out_dir', type=str, default="/dhc/groups/mpws2022cl1/images/heart/png/", help='Directory to save the cleaned csv file')
    args = parser.parse_args()

    # Read in the IIDs to exclude
    to_exclude = pd.read_csv(args.exclude_csv, sep=";")['IID'].tolist()

    # Read the main file and filter out rows that contain an IID in the set of IIDs to exclude
    indiv = pd.read_csv(args.indiv_csv)
    indiv_cleaned = indiv[~indiv['IID'].isin(to_exclude)]

    number_IIDs_indiv = len(indiv)
    number_IIDs_indiv_cleaned = len(indiv_cleaned)
    removed = number_IIDs_indiv - number_IIDs_indiv_cleaned
    print("Removed IIDs: " + str(removed))

    cleaned_csv_name = get_filename(args.indiv_csv)
    cleaned_csv_filepath = os.path.join(args.out_dir, cleaned_csv_name)

    # Write the filtered data to a new CSV file
    indiv_cleaned.to_csv(cleaned_csv_filepath, index=False)

    print("Successfully created: " + cleaned_csv_filepath)

def get_filename(file_path):
    base_name = os.path.basename(file_path)
    return base_name.split('.')[0] + '_cleaned.csv'

if __name__ == '__main__':
    main()

