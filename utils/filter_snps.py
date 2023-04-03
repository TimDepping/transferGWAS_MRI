import argparse
import pandas as pd
import os

'''
This script filters for significant SNPs (output of LMM) with a P-value < 5e-08. 
'''
# create the parser
parser = argparse.ArgumentParser(description='Process imputed PC files.')

# add the argument for the directory path
parser.add_argument('--input_dir', type=str,
                    help='The path to the directory containing the imputed PC files')
parser.add_argument('--output_dir', type=str,
                    help='The path to the output directory')
# parse the arguments
args = parser.parse_args()

# get the directory path from the arguments
input_path = args.input_dir
output_path = args.output_dir

p_value_column = 'P_BOLT_LMM_INF'

# loop over the files in the directory
for filename in os.listdir(input_path):
    if filename.endswith(".imp.txt"):
        file_path = os.path.join(input_path, filename)

        # load the file into a Pandas DataFrame
        df = pd.read_csv(file_path, sep='\t')

        # apply the filter condition
        df2 = df[df[p_value_column] < 5e-08]

        # sort the dataframe by p value in ascending order
        df2 = df2.sort_values(by=p_value_column)

        # select the "SNP" column
        snp_column = df2['SNP']

        # write the "SNP" column to a text file without a header
        output_file = "sig_snps_" + filename[:-4] + ".txt"
        snp_column.to_csv(output_path + output_file, header=False, index=False)
