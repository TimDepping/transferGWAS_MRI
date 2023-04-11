import argparse
import csv

# This script calculates the sum of the explained variance ratio.

def calculate_sum_explained_variance_ratio(filename):
    sum_explained_variance_ratio = 0

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sum_explained_variance_ratio += float(row['explained_variance_ratio'])

    return sum_explained_variance_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='path to the csv file')
    args = parser.parse_args()

    sum_explained_variance_ratio = calculate_sum_explained_variance_ratio(args.filename)

    print("Sum of explained_variance_ratio:", sum_explained_variance_ratio)
