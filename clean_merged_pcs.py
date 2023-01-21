import argparse
import pandas as pd
import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input directory where the PC files are')
    parser.add_argument('--output', type=str, help='Output filename')
    args = parser.parse_args()

    # get a list of all .txt files in the directory
    txt_files = [f for f in os.listdir(args.input) if f.endswith(".txt")]

    # loop through the list of files
    for i, file in enumerate(txt_files):
        # read the file into a pandas dataframe
        df = pd.read_csv(args.input + file, delimiter='\t')
        # store the dataframe with a numerated name
        df.to_csv(args.input+"PC_{}.csv".format(i))

    # get a list of all .csv files in the directory
    csv_files = [f for f in os.listdir(args.input) if f.endswith(".csv")]

    # loop through the list of files
    for file in csv_files:
        # read the file into a pandas dataframe
        df = pd.read_csv(args.input + file)
        # create a new column named 'PC' with the first 4 characters of the file name
        df['PC'] = file[3:4]
        # save the dataframe
        df.to_csv(args.input + file)

    # setting the path for joining multiple files

    filenames = glob.glob(args.input + "/*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename))

    # Concatenate all data into one DataFrame
    df_all = pd.concat(dfs, ignore_index=True)   


    df_all=df_all[['CHR','SNP','BP','GENPOS','ALLELE1','ALLELE0','A1FREQ','BETA','SE','CHISQ_BOLT_LMM_INF','P_BOLT_LMM_INF','PC']]
    df_all[['GENPOS','A1FREQ','BETA','SE','CHISQ_BOLT_LMM_INF','P_BOLT_LMM_INF','PC']] = df_all[['GENPOS','A1FREQ',
    'BETA','SE','CHISQ_BOLT_LMM_INF','P_BOLT_LMM_INF','PC']].apply(pd.to_numeric, errors='coerce')
    df_all[['SNP','PC']] = df_all[['SNP','PC']].astype(str)

    # save the merged dataframe
    df_all.to_csv(args.input + "clean_merged_pc.txt", index=False, sep='\t')

   
if __name__ == '__main__':
    main()