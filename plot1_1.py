import pandas as pd
import matplotlib.pyplot as plt 
import random
import os
import seaborn as sn
import numpy as np

data_dir = 'D:\HMTY\8ο εξάμηνο\Εξόρυξη Δεδομένων\Υλοποιητική Εργασία\harth'
random_file = random.choice(os.listdir(data_dir))
file_path = os.path.join(data_dir, random_file)
with open(file_path,'r') as csvfile: 
    df = pd.read_csv(csvfile)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
fig, axes = plt.subplots(nrows = 2, ncols = 3)

sub1 = df.plot(ax = axes[0][0],x = 'timestamp', y = 'back_x')
df.plot(ax = axes[0][1],x = 'timestamp', y = 'back_y')
df.plot(ax = axes[0][2],x = 'timestamp', y = 'back_z')
df.plot(ax = axes[1][0],x = 'timestamp', y = 'thigh_x')
df.plot(ax = axes[1][1],x = 'timestamp', y = 'thigh_y')
df.plot(ax = axes[1][2],x = 'timestamp', y = 'thigh_z')

plt.suptitle('figure ' + random_file)

std_back_x = np.std(df['back_x'])
std_back_y = np.std(df['back_y'])
std_back_z = np.std(df['back_z'])
std_thigh_x = np.std(df['thigh_x'])
std_thigh_y = np.std(df['thigh_y'])
std_thigh_z = np.std(df['thigh_z'])

mean_back_x = np.mean(df['back_x'])
mean_back_y = np.mean(df['back_y'])
mean_back_z = np.mean(df['back_z'])
mean_thigh_x = np.mean(df['thigh_x'])
mean_thigh_y = np.mean(df['thigh_y'])
mean_thigh_z = np.mean(df['thigh_z'])

median_back_x = np.median(df['back_x'])
median_back_y = np.median(df['back_y'])
median_back_z = np.median(df['back_z'])
median_thigh_x = np.median(df['thigh_x'])
median_thigh_y = np.median(df['thigh_y'])
median_thigh_z = np.median(df['thigh_z'])

print("standard deviation:\nback_x: {:.4f}\nback_y: {:.4f}\nback_z: {:.4f}\nthigh_x: {:.4f}\nthigh_y: {:.4f}\nthigh_z: {:.4f}\n". 
      format(float(std_back_x),float(std_back_y),float(std_back_z),float(std_thigh_x),float(std_thigh_y),float(std_thigh_z)))

print("mean:\nback_x: {:.4f}\nback_y: {:.4f}\nback_z: {:.4f}\nthigh_x: {:.4f}\nthigh_y: {:.4f}\nthigh_z: {:.4f}\n". 
      format(float(mean_back_x),float(mean_back_y),float(mean_back_z),float(mean_thigh_x),float(mean_thigh_y),float(mean_thigh_z)))

print("median:\nback_x: {:.4f}\nback_y: {:.4f}\nback_z: {:.4f}\nthigh_x: {:.4f}\nthigh_y: {:.4f}\nthigh_z: {:.4f}\n". 
      format(float(median_back_x),float(median_back_y),float(median_back_z),float(median_thigh_x),float(median_thigh_y),float(median_thigh_z)))

plt.figure()
corr_matrix = df.loc[:, ~df.columns.isin(['label','timestamp'])].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
ax = sn.heatmap(
    corr_matrix,
    annot = True,
    vmax = 1,
    vmin = -1,
    center = 0,
    cmap = 'vlag',
    mask = mask
)

plt.suptitle('figure ' + random_file)

concat = pd.DataFrame()
for root, _, files in os.walk(data_dir):
    for file in files:
        new = pd.read_csv(os.path.join(root, file), index_col = 'timestamp')
        concat = pd.concat([concat, new])
concat.drop(labels = ['index', 'Unnamed: 0'], axis = 'columns', inplace = True)
concat.reset_index(inplace = True)
concat['timestamp'] = pd.to_datetime(concat['timestamp']).dt.time

plt.figure()
corr_concat_matrix = concat.loc[:, ~concat.columns.isin(['label','timestamp'])].corr()
mask = np.triu(np.ones_like(corr_concat_matrix, dtype = bool))
ax1 = sn.heatmap(
    corr_concat_matrix,
    annot = True,
    vmax = 1,
    vmin = -1,
    center = 0,
    cmap = 'vlag',
    mask = mask
)
plt.suptitle('concatenated data figure')

plt.show()
