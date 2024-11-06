# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:01:12 2023

@author: JinYu
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import statsmodels.stats.multitest as multi

import seaborn as sns
import matplotlib.pyplot as plt

#####Calculate Fisher's exact test

"""
---------------------------------------------
case_id | group | gene_1 | gene_2 | ... |
R01-001 | short |  10.1  |  12.3  | ... |
R01-002 | long  |  12.1  |  99.3  | ... |
---------------------------------------------
"""
path = "D:/pathological/DEA/response/dataset/missf_training.csv"
print(path)
data = pd.read_csv(path ,header = 0,index_col = 0)
genename = list(data.columns.values)[1:]
print(genename)
casename = list(data.index.values)
#print(casename)
genenumber = len(genename)
print(genenumber)
savepath = "D:/pathological/DEA/response/result/missf0911.csv"

# Create 2X2 contengency table
gene_long=[]
gene_short=[]
fold_change=[]
for gene in genename:
    group0 = data.loc[data.loc[:,'RT_response']=='Non-response'][gene] #baseline
    group1 = data.loc[data.loc[:,'RT_response']=='Response'][gene] #positive 分子
    long = group0.mean()
    short = group1.mean()
    fold=short/long
    foldchange = np.log2(fold)    #take log2
    fold_change.append(foldchange)
    gene_test_long = group0.sum()
    gene_test_short = group1.sum()
    gene_long.append(gene_test_long)
    gene_short.append(gene_test_short)
    remain_gene_long = np.zeros(genenumber)
    remain_gene_short = np.zeros(genenumber)
#print("creating 2x2 contengency table...")
for i in range(genenumber):
    x=0 #long
    y=0 #short
    for gene in genename:
        if gene != genename[i]:
            for case in casename:
                j=0
                if data.loc[case,'RT_response']=='Non-response':   # 分母
                    x+=data[gene][case]
                    j = j+1
                else:
                    y+=data[gene][case]
                    j = j+1
    remain_gene_long[i]=x
    remain_gene_short[i]=y


#print(gene_long)   
#print(gene_short)
#print(remain_gene_long)
#print(remain_gene_short)

p_val = []
print("calculating p_value ...")
for i in range(genenumber):
    #print(i)
    table = pd.DataFrame(columns=["long","short"],index=["gene test","gene renamin"])
    table.loc["gene test","long"] = gene_long[i]
    table.loc["gene test","short"] = gene_short[i]
    table.loc["gene renamin","long"] = remain_gene_long[i]
    table.loc["gene renamin","short"] = remain_gene_short[i]
    table = np.array([[gene_long[i],gene_short[i]],[remain_gene_long[i],remain_gene_short[i]]])
    #print(table)
    fishertest = fisher_exact(table, alternative='two-sided')
    #print(fishertest)
    p_val.append(fishertest[1])
#print(p_val)


myarray = np.asarray(p_val)
foldchange = np.asarray(fold_change)
result = pd.DataFrame({'pvalue':myarray,'FoldChange_log2':foldchange})
result['log(pvalue)'] = -np.log10(result['pvalue'])
result['expression'] = 'Normal'

cutoff = 0.01

result.loc[(result.FoldChange_log2 > cutoff )&(result.pvalue < 0.05),'expression'] = 'Over'    # cutoff -- 1
result.loc[(result.FoldChange_log2 < -cutoff )&(result.pvalue < 0.05),'expression'] = 'Low'


pv = result['pvalue']

pval_corr = multi.multipletests(pv, alpha=0.1, method='fdr_bh')
# alpha: FWER, family-wise error rate;  method: Benjamini Hochberg correction
# https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
result['adjust_p']=pval_corr[1]
result['log(adjust_p)'] = -np.log10(result['adjust_p'])
result['expression-adjust-bh'] = 'Normal'
result.loc[(result.FoldChange_log2 > cutoff )&(result.adjust_p < 0.05),'expression-adjust-bh'] = 'Over'   # cutoff -- 1
result.loc[(result.FoldChange_log2 < -cutoff )&(result.adjust_p < 0.05),'expression-adjust-bh'] = 'Low'


result.index = genename
result.to_csv(savepath)

print("finished")


"""
-----------------------------------------------------------------
feature | pvalue | log(pvalue) | FoldChange_log2 | expression |
-----------------------------------------------------------------
"""

"""
plt.figure(figsize=(6,5))
ax = sns.scatterplot(x="logHR", y="log(pvalue)",
                     hue_order = ('Low','Normal','Over'), hue='expression',
                     palette=("#87CEFA","#000000","#B22222"), s=10,
                     data=result)

ax.axis([-6.5,6.5,0,7])
#ax.axvline(1.5, 0,10,linestyle="--",alpha=0.55,color='grey')
#ax.axvline(-1.5, 0,10,linestyle="--",alpha=0.55,color='grey')
ax.axvline(0, 0,10,linestyle="--",alpha=0.55,color='grey')


ax.axhline(1.3, -1,6,linestyle="--", alpha=0.55,color='grey')
#ax.text(-6,10.1,"Low Expression <--")
#ax.text(2.9,10.1,"--> Over Expression")
#ax.text(-1.2,-0.21,"-1")
#ax.text(0.89,-0.21,"1")
ax.text(-7.1,1.25,"1.3")
ax.text(-7.5,2.9,"10",fontsize=9,rotation=90)
ax.text(-1.7,-0.9,"2",fontsize=9)

#ax.set_ylabel('Feature value (normalized with Z-Score)',fontsize=11)
ax.set_ylabel('-log   (p-value)',fontsize=16)
ax.set_xlabel('log  (FoldChange)',fontsize=16)  #fontweight='bold',
#ax.set_title('Volcano Plot of DEA',y=1,fontweight='bold',fontsize=15)
#ax.figure.savefig('C:/Users/ariken/Desktop/dea.png',dpi=800,bbox_inches='tight')
"""