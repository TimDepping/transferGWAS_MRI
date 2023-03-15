import os
import pandas as pd
import argparse
from plots import qq_plot, mhat_plot

pval_name = 'P_BOLT_LMM_INF'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        help='Path to lmm results directory.')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.results_dir):
        f = os.path.join(args.results_dir, filename)
        if os.path.isfile(f):
            name = os.path.splitext(os.path.basename(f))[0]
            print(f'Create plots for: {name}')
            try:
                df = pd.read_csv(f, sep="\t")
                qq_plot(df[pval_name], fn=f'{args.output_dir}/{name}_qq.png')
                mhat_plot(df, pval_name, fn=f'{args.output_dir}/{name}_mhat')
            except:
                print(
                    f'An exception occurred while creating the plot for: {name}')


if __name__ == "__main__":
    main()
