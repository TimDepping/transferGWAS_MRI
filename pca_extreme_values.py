import pandas as pd
import numpy as np
import seaborn as sns
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Input resnet50_imagenet_L4 file')
parser.add_argument('--pca', type=str, default= '0', help='Select the PCA')


args = parser.parse_args()


pc_resnet= pd.read_csv(args.input, delim_whitespace=True)

print(pc_resnet.columns)
print(pc_resnet.index)

#Extract the first 10 smallest PCA values
pc_0= pc_resnet[['FID','IID','PC_'+args.pca]]
pc_0.sort_values( by= 'PC_'+args.pca, ascending=True, inplace=True)
pc_0_min= pc_0.nsmallest(10, columns= 'PC_'+args.pca, keep='first')
print(pc_0_min)
pc_0_min.to_csv('PC_'+args.pca + '_min_pca.csv')

#Extract the first 10 smallest PCA values
pc_0= pc_resnet[['FID','IID','PC_'+args.pca]]
pc_0.sort_values( by= 'PC_'+args.pca,ascending=False, inplace=True)
pc_0_max= pc_0.nlargest(10, columns= 'PC_'+args.pca, keep='first')
pc_0_max.to_csv('PC_'+args.pca + 'max_pca.csv')
