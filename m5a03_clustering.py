import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans


def do_kmeans(data, clusters=1):
    model = KMeans(n_clusters=clusters)
    model.fit(data[['TowerLon', 'TowerLat']])
    return model


def find_min_cluster(model):
    count = {}
    for i in model.labels_:
        if i in count.keys():
            count[i] += 1
        else:
            count[i] = 1
    count_inv = {v: k for k, v in count.items()}
    min_cluster = count_inv[min(count.values())]
    print(f"\tCluster with fewest samples: {min_cluster}")
    return model.labels_ == min_cluster


def cluster_info(model):
    print(f"\tCluster Analysis Inertia: {model.inertia_}")
    unique, counts = np.unique(model.labels_, return_counts=True)
    for i in range(len(unique)):
        print("\t" + "-" * 9)
        print(f"\tCluster {unique[i]}")
        print(f"\tCentroid {model.cluster_centers_[i]}")
        print(f"\t#Samples {counts[i]}")
    print("\t" + "-" * 9)


# Load up the dataset and take a peek at its head and dtypes:
df = pd.read_csv('datasets/call_detail_records.csv',
                 dtype={'In': str, 'Out': str})
print(df.head())
print(f"\nOld dtypes:\n{df.dtypes}")
# Convert the date and the time:
df['CallDate'] = pd.to_datetime(df['CallDate'], errors='coerce')
df['Duration'] = pd.to_timedelta(df['Duration'], errors='coerce')
df['CallTime'] = pd.to_timedelta(df['CallTime'], errors='coerce')
print(f"\nNew dtypes:\n{df.dtypes}")
# Distinct list of "In" phone numbers:
users = df['In'].drop_duplicates().values
# Create a slice of the first user in the dataset:
user1 = df[df['In'] == users[0]]
print(f"\nExamining person: {users[0]}")
# Filter examining only weekdays before 5pm:
user1 = user1[((user1['DOW'] != 'Sat') &
               (user1['DOW'] != 'Sun')) &
              (user1['CallTime'] < '17:00:00')]
# Run KMeans with k=4:
model1 = do_kmeans(user1, 4)
cluster_info(model1)
# Find the cluster with the least attached nodes:
midway_samples = user1[find_min_cluster(model1)]
t_sec = round(midway_samples['CallTime'].mean().total_seconds())
print("\tIts waypoint time: {}".format(dt.time(hour=t_sec // 3600,
                                               minute=t_sec % 3600 // 60,
                                               second=t_sec % 3600 % 60)))
# Plot the Cell Towers the user connected to:
fig = plt.figure()
ax = fig.add_subplot(111)
print(f"\tWeekday Calls (<5pm): {len(user1)}")
ax.scatter(user1['TowerLon'], user1['TowerLat'],
           c=model1.labels_, marker='o', edgecolor='black', alpha=.8)
# Draw the centroids for the clusters:
ax.scatter(model1.cluster_centers_[:, 0], model1.cluster_centers_[:, 1],
           s=169, c='r', marker='x')
# Show graph:
ax.set_title(f'Weeday Calls (<5pm) for user {users[0]}')
plt.show()
