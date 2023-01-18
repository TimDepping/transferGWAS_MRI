import pandas as pd
import argparse
import os
import shutil

def copy_image(iid, source_dir, output_dir):
    files_with_iid = [filename for filename in os.listdir(source_dir) if filename.startswith(str(iid))]
    if len(files_with_iid) == 1:
        first_filename = files_with_iid[0]
        output_file_path = f'{output_dir}/{iid}.png'
        if os.path.isfile(output_file_path):
            output_file_path = f'{output_dir}/{iid}_dup.png'
        shutil.copyfile(f'{source_dir}/{first_filename}', output_file_path)
    elif len(files_with_iid) > 1:
        print("There exists multiple files for this iid.")
    else:
        print(f'There is no file starting with idd: {iid}.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Textfile which includes principal components for each individual e.g. resnet50_imagenet_L4.txt')
    parser.add_argument('--output_dir', type=str, default="./out", help='Directory to save the image files')
    parser.add_argument('--n_extreme', type=int, default= '10', help='Number of extreme values to extract')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, delim_whitespace=True)
    df.drop_duplicates(subset='IID', keep='last', inplace=True)
    pc_columns = [column for column in df if column.startswith('PC')]

    extreme_values_df = pd.DataFrame(columns=['IID','PC','PC_VALUE'])
    for pc_column in pc_columns:
        pc_df = df[['IID', pc_column]]
        pc_df = pc_df.sort_values(by= pc_column, ascending=True)
        pc_min = pc_df.nsmallest(args.n_extreme, columns=pc_column)
        pc_max = pc_df.nlargest(args.n_extreme, columns=pc_column)
        pc_extreme = pd.concat([pc_min, pc_max])
        pc_output_dir=f'{args.output_dir}/{pc_column}'
        os.makedirs(pc_output_dir, exist_ok=True)
        for _, row in pc_extreme.iterrows():
            iid = str(int(row['IID']))
            pc_value=str(row[pc_column])
            extreme_values_df.loc[len(extreme_values_df)]=[iid, pc_column, pc_value] 
            copy_image(iid=iid,
            source_dir="/dhc/groups/mpws2022cl1/images/heart/png/50000_RGB_0-16-39",
            output_dir=pc_output_dir)
    extreme_values_df.to_csv(f'{args.output_dir}/extreme_ids.csv', index=False)

if __name__ == "__main__":
    main()