import pandas as pd
import argparse

ej_column = '22420-2.0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str,
                        help='Path to file to extract the IDs from')
    parser.add_argument('--label_file', type=str,
                        help='Path to file to extract the values for the label e.g. ejection fraction')
    parser.add_argument('--output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()

    print("here")
    embeddings_df = pd.read_csv(args.embeddings_file, sep=' ')
    label_df = pd.read_csv(args.label_file, sep=',')
    print(label_df)
    print(label_df[ej_column])
    merged_df = embeddings_df.merge(label_df, left_on='IID', right_on='eid')
    merged_df = merged_df[['FID', 'IID', ej_column]]
    merged_df.dropna(subset=[ej_column], inplace=True)
    merged_df = merged_df.rename(columns={ej_column: 'PC_0'})
    print(merged_df)
    merged_df.to_csv(args.output_file, sep=' ', index=False)


if __name__ == "__main__":
    main()
