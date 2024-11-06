# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:34:03 2023

@author: ariken
"""



#%%
################## feature collection #####################

import os
import pandas as pd
import numpy as np

luadpath = ".csv"
#luscpath = ".csv"

luad = pd.read_csv(luadpath, index_col=0, header=0)
#lusc = pd.read_csv(luscpath, index_col=0, header=0)

imgname = luad.index.values
featurename = luad.columns.values
color = ["norm","hematoxylin","eosin"]
channel = ['R','G','B']

counts = int(len(imgname)/9)

new_feature = []
new_index = []
patient_feature = []

for co in color:
    for cha in channel:
        for feature in featurename:
            newfeaturename = co + '_' + cha + '_' + feature
            new_feature.append(newfeaturename)

for i in range (counts):
    text = str(imgname[9*i])
    patientid, colo, channe = text.split('_')
    new_index.append(patientid)
    
    data = luad[imgname[i*9]:imgname[i*9+8]]
    data_reshaped = data.values.reshape(-1)
    val = np.array(data_reshaped)
    patient_feature.append(val)
    
    
# data = luad[imgname[0]:imgname[8]]
# data_reshaped = data.values.reshape(-1,1)



# data1 = luad[imgname[18]:imgname[26]]
    
    
    
result = pd.DataFrame(patient_feature,index=new_index,columns=new_feature)
result.to_csv("lusc_feature_fin.csv")



#%%


#################change id name###################
import os
import pandas as pd
import numpy as np


path = "lusc1030.csv"

df = pd.read_csv(path, index_col=0, header=0)

img_name = df.index.values

imgid = []

for img in img_name:
    file_path, file_name = os.path.split(img)
    text = file_path.split('_')[2]
    patientid = ''.join(list(text)[5:17])
    idname = patientid + '_' + os.path.splitext(file_name)[0]
    imgid.append(idname)

df['patient']=imgid

df.to_csv("lusc.csv")


#%%

################## mean value #####################

import os
import pandas as pd
import numpy as np

path = "lusc.csv"
df = pd.read_csv(path, index_col=0, header=0)

start = 0
end = 5

imgname = df.index.values
featurename = df.columns.values

counts = int(len(imgname)/5)

feature_mean = []
new_index = []

for i in range(counts):
    # print((start+5)*i)
    # print(end*(i+1))
    # print("--------")
       
    data = df[(start+5)*i:end*(i+1)]
    mean_val = np.array(data.mean())
    feature_mean.append(mean_val)
    text = str(imgname[i*5])
    patientid, tile, color, channel = text.split('_')
    fin_name = patientid + '_' + color + '_' + channel
    new_index.append(fin_name)
    
result = pd.DataFrame(feature_mean,index=new_index,columns=featurename)
result.to_csv("lusc_feature.csv")




