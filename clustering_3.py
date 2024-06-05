import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


'''
Activities:

1: walking	
2: running	
3: shuffling
4: stairs (ascending)	
5: stairs (descending)	
6: standing	
7: sitting	
8: lying	
13: cycling (sit)	
14: cycling (stand)	
130: cycling (sit, inactive)
140: cycling (stand, inactive)
'''

data_dir = './Υλοποιητική Εργασία/harth'

concat = pd.DataFrame()
for root, _, files in os.walk(data_dir):
    for file in files:
        new = pd.read_csv(os.path.join(root, file), index_col = 'timestamp')
        file_name = os.path.splitext(file)[0]
        new['file_name'] = file_name
        concat = pd.concat([concat, new])
concat.drop(labels = ['index', 'Unnamed: 0', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'], axis = 'columns', inplace = True)
concat.reset_index(inplace = True)
concat['timestamp'] = pd.to_datetime(concat['timestamp']).dt.time

#print(concat) 

user_activity_counts = concat.groupby('file_name')['label'].nunique().reset_index()

# Using KMeans to cluster users based on activity counts
scaler = StandardScaler()
X = scaler.fit_transform(user_activity_counts[['label']])
kmeans = KMeans(n_clusters=3, random_state=42)
user_activity_counts['cluster'] = kmeans.fit_predict(X)

# Visualizing the clusters
plt.figure(figsize=(8, 6))
for cluster_label in user_activity_counts['cluster'].unique():
    cluster_data = user_activity_counts[user_activity_counts['cluster'] == cluster_label]
    plt.scatter(cluster_data['file_name'], cluster_data['label'], label=f'Cluster {cluster_label + 1}')

plt.xlabel('User')
plt.ylabel('Number of Activities')
plt.title('User Clustering based on Activity Counts')
plt.xticks(rotation=45)
plt.legend()
plt.show()