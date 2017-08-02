import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def do_kmeans(data, k=7):
    # Plot your data at the Longitude and Latitude locations:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data['Longitude'], data['Latitude'], marker='.', alpha=0.3)
    # Use K-Means to try and find seven cluster centers:
    model = KMeans(n_clusters=k)
    model.fit(data[['Longitude', 'Latitude']])
    # Print and plot the centroids:
    centroids = model.cluster_centers_
    print(f"Centroids:\n{centroids}")
    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker='x', c='red', alpha=0.5, linewidths=3, s=169)


# Load the dataset:
df = pd.read_csv('datasets/gambling_crimes.csv')
# Drop any rows with NaNs:
df.dropna(inplace=True)
# Print out the dtypes:
print(df.dtypes)
# Coerce the 'Date' feature into real date:
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print(df.dtypes)
# Use kmeans on data:
do_kmeans(df)
# Use kmenas with filtered data:
do_kmeans(df[df['Date'] > '2011-01-01'])
plt.savefig('plots/kmeans.png')
# Show graphs:
plt.show()
