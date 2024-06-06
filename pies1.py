import pandas as pd
import matplotlib.pyplot as plt 
import os
import seaborn as sn
import numpy as np
import glob
import datetime


# Φόρτωση όλων των αρχείων .csv στον φάκελο 'harth'
files = glob.glob('harth/*.csv')


# Φορτώστε τα δεδομένα σας
ctr = 0
dataframes = [pd.read_csv(file) for file in files]
for df in dataframes:
    ctr+=1
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.timestamp())
all_df = pd.concat(dataframes)



Activities={
1: 'walking',	
2: 'running',	
3: 'shuffling',
4: 'stairs (ascending)',	
5: 'stairs (descending)',
6: 'standing',	
7: 'sitting',	
8: 'lying',	
13: 'cycling (sit)',	
14: 'cycling (stand)',	
130: 'cycling (sit, inactive)',
140: 'cycling (stand, inactive)'
}

all_sizes = np.zeros(len(Activities))

ctr = 0
for individual_data in dataframes:
    x = np.array(individual_data['timestamp'])
    y = np.array(individual_data['label'])
    labels= []
    sizes = []
    
    if (ctr%6==0):
        fig, axs = plt.subplots(2, 3)
    for act in Activities:
        x_act = x[np.where(y == act)]
        seconds = len(x_act)/50
        minutes = seconds/60
        labels.append(Activities[act])
        sizes.append(minutes)
    axs.flat[ctr%6].title.set_text('{}'.format(ctr))
    axs.flat[ctr%6].pie(sizes)
    ctr+=1
    all_sizes = all_sizes + np.array(sizes)

plt.figure()
plt.pie(all_sizes)
plt.legend(labels, bbox_to_anchor =(-0.3,-0.1), loc = 'lower left', prop = {'size' : 8})
plt.show()
