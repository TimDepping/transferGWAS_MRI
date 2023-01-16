import pandas as pd
import matplotlib.pyplot as plt
import geneview as gv

pc0= pd.read_csv('PC_0.csv')
pc1= pd.read_csv('PC_1.csv')
pc2= pd.read_csv('PC_2.csv')
pc3= pd.read_csv('PC_3.csv')
pc4= pd.read_csv('PC_4.csv')
pc5= pd.read_csv('PC_5.csv')
pc6= pd.read_csv('PC_6.csv')
pc7= pd.read_csv('PC_7.csv')
pc8= pd.read_csv('PC_8.csv')
pc9= pd.read_csv('PC_9.csv')

frames= [pc0,pc1,pc2,pc3,pc4,pc5,pc6,pc7,pc8,pc9]

print(pc0.shape)
print(pc1.shape)

pc0[['pc']]=0
pc1[['pc']]=1
pc2[['pc']]=2
pc3[['pc']]=3
pc4[['pc']]=4
pc5[['pc']]=5
pc6[['pc']]=6
pc7[['pc']]=7
pc8[['pc']]=8
pc9[['pc']]=9

pc_total= result = pd.concat(frames)

pc_total
df= pc_total.to_csv("pc_total.csv", index=None)

df2= pc_total[['CHR','GENPOS','ALLELE1','ALLELE0','A1FREQ','BETA','SE','CHISQ_LINREG','P_LINREG']]

df2.rename(columns={'CHR': '#CHROM', 'GENPOS': 'POS','ALLELE1': 'A1','ALLELE0': 'ALT','CHISQ_LINREG': 'T_STAT',
                   'P_LINREG': 'P'}, inplace=True)

#CHISQ_BOLT_LMM_INF	P_BOLT_LMM_INF
df3= pc_total[['CHR','GENPOS','ALLELE1','ALLELE0','A1FREQ','BETA','SE','CHISQ_BOLT_LMM_INF','P_BOLT_LMM_INF']]

df3.rename(columns={'CHR': '#CHROM', 'GENPOS': 'POS','ALLELE1': 'A1','ALLELE0': 'ALT','CHISQ_BOLT_LMM_INF': 'T_STAT',
                   'P_BOLT_LMM_INF': 'P'}, inplace=True)

ax = gv.manhattanplot(data=df3)
plt.title('Manhattan plot /BOLT LMM')
plt.figure(figsize=(25, 10))
plt.savefig('PC_9.png', dpi=300, bbox_inches='tight')


plt.show()

df_fin= pc_total.sort_values('P_BOLT_LMM_INF', ascending=True)

df_fin2=df_fin.head(30)

df_fin2

df_fin2.to_csv('pc_top30.csv')

