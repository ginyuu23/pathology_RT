# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:09:32 2023

@author: JinYu
"""

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import plot_interval_censored_lifetimes
import pandas as pd
import matplotlib.font_manager
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings(action='ignore')

#df = pd.read_csv("kmtest.csv",header=0,index_col=0)
df = pd.read_csv("./km_test.csv",header=0,index_col=0)
#df = pd.read_excel("./clinical_all.xlsx", sheet_name= "all_checked", header=0,index_col=0)



kmf = KaplanMeierFitter(alpha = 0.95)   #alpha设置置信区间

#lifelines.plotting.plot_interval_censored_lifetimes

T = df['time'] / 365                       #生存时长（天数）
E = df['SurvivalStatus']  
groups = df['missfall_lsvm']  #baseline=logFC2
ix = (groups == 'Response')

available_fonts = matplotlib.font_manager.findSystemFonts()
print(available_fonts)  # This will print a list of available font files

# Choose a font from the list and set it
chosen_font = 'Century'  # For example, you can choose Arial if available
plt.rcParams['font.family'] = chosen_font

plt.figure(figsize=(6,4.7),dpi=500)

kmf.fit(T[~ix], E[~ix], label='Non-response')
ax = kmf.plot_survival_function()  #有置信区间
kmf.fit(T[ix], E[ix], label='Response')
ax = kmf.plot_survival_function(ax=ax)
ax.tick_params(direction='in')
plt.xlabel('Time since diagnosis (year)')
plt.ylabel('Overall survival')

#plt.text(-1.2,1.1,"d",fontweight='bold',fontsize=20)
plt.title('Test patients (n=22)',fontweight='bold',fontsize=13)
#plt.text(-1.5,1.04,"b",fontweight='bold',fontsize=18)
plt.xlim([0,7])
plt.ylim([0, 1.05])


T_exp, E_exp = df.loc[ix, 'time'], df.loc[ix, 'SurvivalStatus']
T_con, E_con = df.loc[~ix, 'time'], df.loc[~ix, 'SurvivalStatus']
results = logrank_test(T_exp, T_con, event_observed_A=E_exp, event_observed_B=E_con)

p_value = results.p_value
plt.text(0.2,0.05,f'p(log-rank test)={p_value:.4f}')
#plt.text(4,0.25,'p(log-rank test)=1.24 x 10$^{-7}$')


results.print_summary()
print(results.p_value)      
