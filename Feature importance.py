# -*- coding: utf-8 -*-
"""
Created on Thu May  2 00:38:18 2019

@author: sr4376
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
FIMain = None
years=np.arange(2008,2016)
for ii in years:
    FI1 =pd.read_csv(r'C:\comp\FI1' + str(ii) + '.csv')
    FI1.rename(columns={'Unnamed: 0':'features','0':ii},inplace=True)
    FI1.set_index('features',inplace=True)
    if FIMain is None:
        FIMain=FI1
    else:
        FIMain=pd.merge(FIMain,FI1,how='left',left_index=True,right_index=True)
        
        
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = FIMain.columns.values
farmers = FIMain.columns.values

harvest = np.array(FIMain)


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Features overTime")
fig.tight_layout()
plt.show()