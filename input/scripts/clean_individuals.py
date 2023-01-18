import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indiv_csv', type=str, help='Path to csv file holding IID and the path to image e.g. 1000_RGB_0-16-39.csv')
    parser.add_argument('exclude_csv', type=str, help='Path to csv file holding IIDs to exclude')
    parser.add_argument('output_csv', type=str, default="./out/cleaned_indiv.csv", help='Path to save the cleaned csv file')
    args = parser.parse_args()

    # Read in the IIDs to exclude
    IIDs_to_exclude = pd.read_csv(args.exclude_csv)['IID'].tolist()
    # Read the main file and filter out rows that contain an IID in the set of IIDs to exclude
    indiv = pd.read_csv(args.indiv_csv, sep=",")
    indiv_cleaned = indiv[~indiv['IID'].isin(IIDs_to_exclude)]

    # Drop IIDs which appear multiple times and only keep the last occurence
    indiv_cleaned = indiv_cleaned.drop_duplicates(subset=indiv.columns[0],keep='last')

    number_IIDs_indiv = len(indiv)
    number_IIDs_indiv_cleaned = len(indiv_cleaned)
    removed = number_IIDs_indiv - number_IIDs_indiv_cleaned
    print("Removed IIDs: " + str(removed))

    # Write the filtered data to a new CSV file
    indiv_cleaned.to_csv(args.output_csv, index=False)

    print("Successfully created: " + args.output_csv)

if __name__ == '__main__':
    main()

