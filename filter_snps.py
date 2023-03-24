import argparse
import pandas as pd
import os

'''
This script filters for significant SNPs (output feature condensation) with a P-value < 5e-08. 
'''
# create the parser
parser = argparse.ArgumentParser(description='Process imputed PC files.')

# add the argument for the directory path
parser.add_argument('--path', type=str, help='The path to the directory containing the imputed PC files')
parser.add_argument('--output_dir', type=str, help='The path to the output directory')
# parse the arguments
args = parser.parse_args()

# get the directory path from the arguments
dir_path = args.path
output_path = args.output_dir

# loop over the files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".imp.txt"):
        file_path = os.path.join(dir_path, filename)

        # load the file into a Pandas DataFrame
        df = pd.read_csv(file_path, sep='\t')

        # apply the filter condition
        df2 = df[df['P_BOLT_LMM_INF'] < 5e-08]

        # sort the dataframe by "P_BOLT_LMM_INF" in ascending order
        df2 = df2.sort_values(by='P_BOLT_LMM_INF')

        # select the "SNP" column
        snp_column = df2['SNP']

        # write the "SNP" column to a text file without a header
        output_file = "sig_snps_" + filename[:-4] + ".txt"
        snp_column.to_csv(output_path + output_file, header=False, index=False)
