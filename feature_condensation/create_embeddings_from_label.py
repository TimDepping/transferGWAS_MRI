import pandas as pd
import argparse

# This script takes two csv files: 1. A csv file which contains the individuals with their PCs. 2. A csv file which holds the labels for all individuals of the UKBiobank.
# The script creates a new csv file that contains the columns: IID, FID and PC_0. The column PC_0 contains the value of the ejection fraction.
# The column is named 'PC_0' so that the embeddings file can directly be used in feature_condensation.py script.
# Only the individuals of the first csv file are included.

ej_column = '22420-2.0'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embeddings_file', type=str,
                        help='Path to file to extract the IDs from')
    parser.add_argument('-l', '--label_file', type=str,
                        help='Path to file to extract the values for the label e.g. ejection fraction')
    parser.add_argument('-o', '--output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()

    embeddings_df = pd.read_csv(args.embeddings_file, sep=' ')
    label_df = pd.read_csv(args.label_file, sep=',')
    merged_df = embeddings_df.merge(label_df, left_on='IID', right_on='eid')
    merged_df = merged_df[['FID', 'IID', ej_column]]
    merged_df.dropna(subset=[ej_column], inplace=True)
    merged_df = merged_df.rename(columns={ej_column: 'PC_0'})
    print(merged_df)
    merged_df.to_csv(args.output_file, sep=' ', index=False)


if __name__ == "__main__":
    main()
