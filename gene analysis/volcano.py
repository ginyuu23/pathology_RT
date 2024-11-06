# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:20:23 2023

@author: JINYU
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("missf0911.csv",index_col=0,header=0)


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.figure(figsize=(6,5),dpi=800) #, facecolor='none', edgecolor='none'
ax = sns.scatterplot(x="FoldChange_log2", y="log(pvalue)",
                     hue_order = ('Low','Normal','Over'), hue='expression',
                     palette=("#87CEFA","#808080","#B22222"), s=30,
                     data=df)


plt.legend(ncol=3, loc='upper center',fontsize=11)
plt.tick_params(direction='in')
# ax.figure.set_facecolor('none')
# ax.set_facecolor('none')

# add gene name
threshold_pvalue = 1.3
threshold_fold_change = 0  

for i, row in df.iterrows():
    if row['log(pvalue)'] > threshold_pvalue and (row['FoldChange_log2'] > threshold_fold_change or row['FoldChange_log2'] < -threshold_fold_change):
        ax.annotate(i, (row['FoldChange_log2']+0.01, row['log(pvalue)']+0.05), fontsize=9, alpha=0.7)



ax.axis([-0.35,0.35,0,9])
#ax.axvline(1.5, 0,10,linestyle="--",alpha=0.55,color='grey')
#ax.axvline(-1.5, 0,10,linestyle="--",alpha=0.55,color='grey')
ax.axvline(0.05, 0,10,linestyle="--",alpha=0.55,color='grey', linewidth=0.8)
ax.axvline(-0.05, 0,10,linestyle="--",alpha=0.55,color='grey', linewidth=0.8)


ax.axhline(1.3, -5,6,linestyle="--", alpha=0.55,color='grey', linewidth=0.8)
#ax.text(-6,10.1,"Low Expression <--")
#ax.text(2.9,10.1,"--> Over Expression")
#ax.text(-1.2,-0.21,"-1")
#ax.text(0.89,-0.21,"1")
ax.text(-0.33,1.4,"-log$_{10}$(0.05)",fontsize=8, alpha=0.7)
# ax.text(-0.31,1.85,"10",fontsize=9,rotation=90)
# ax.text(-0.07,-0.6,"2",fontsize=9)

#ax.set_ylabel('Feature value (normalized with Z-Score)',fontsize=11)
ax.set_ylabel('-log$_{10}$(p-value)',fontsize=13)
ax.set_xlabel('log$_{2}$(Fold change)',fontsize=13)  #fontweight='bold',
#ax.set_title('Volcano Plot of RS genes -- missf',y=1,fontweight='bold',fontsize=15)
#ax.figure.savefig("B1feature_volcano_zoom.png",dpi=800,bbox_inches='tight')

