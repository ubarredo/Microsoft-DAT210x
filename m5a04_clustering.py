import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def do_kmeans(data, clusters=1):
    model = KMeans(n_clusters=clusters)
    model.fit(data)
    return model.cluster_centers_, model.labels_


def do_pca(data, dimensions=2):
    model = PCA(n_components=dimensions, svd_solver='randomized',
                random_state=7)
    model.fit(data)
    return model


# See original features in P.C.Space?
plot_vectors = True
# Colors:
plt.style.use('ggplot')
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']
# Load up the dataset:
df = pd.read_csv('datasets/wholesale_customers.csv')
# Set NaNs to zero:
df.fillna(0, inplace=True)
# Get rid of the 'Channel' and 'Region' columns:
df.drop(['Channel', 'Region'], axis=1, inplace=True)
# Describe data and plot hist:
print(f"Describe:\n{df.describe()}")
df.plot.hist(stacked=True, title='Stacked Hist')
# Remove top 5 and bottom 5 samples for each column:
drop = set()
for col in df.columns:
    sort = df.sort_values(by=col)
    drop.update(list(sort[:5].index) + list(sort[-5:].index))
print(f"\nDropping {len(drop)} Outliers...")
df.drop(labels=drop, axis=0, inplace=True)
print(f"New Describe:\n{df.describe()}")
# Select processing method:
choice = 4
if choice == 1:
    processed = preprocessing.StandardScaler().fit_transform(df)
elif choice == 2:
    processed = preprocessing.MinMaxScaler().fit_transform(df)
elif choice == 3:
    processed = preprocessing.MaxAbsScaler().fit_transform(df)
elif choice == 4:
    processed = preprocessing.Normalizer().fit_transform(df)
else:
    processed = df
# Do KMeans:
centroids, labels = do_kmeans(processed, clusters=6)
# Print out your centroids in feature-space:
print("\nCentroids:")
for i in centroids:
    print(f"\t{i}")
# Do PCA:
display_pca = do_pca(processed)
# Project samples and the centroids into the new 2D feature space:
p_pca = display_pca.transform(processed)
c_pca = display_pca.transform(centroids)
# Visualize all the samples:
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the index of the sample:
for i in range(len(p_pca)):
    ax.text(p_pca[i, 0], p_pca[i, 1], df.index[i],
            color=c[labels[i]], alpha=.8)
    ax.set_xlim(min(p_pca[:, 0]) * 1.2, max(p_pca[:, 0]) * 1.2)
    ax.set_ylim(min(p_pca[:, 1]) * 1.2, max(p_pca[:, 1]) * 1.2)
# Plot the centroids by label:
for i in range(len(centroids)):
    ax.text(c_pca[i, 0], c_pca[i, 1], str(i), fontsize=25, zorder=1000)
# Scale the principal components to estimate the variance:
xvector = display_pca.components_[0] * max(p_pca[:, 0]) / 2
yvector = display_pca.components_[1] * max(p_pca[:, 1]) / 2
# Show the importance of each feature:
importance = sorted([(math.sqrt(xvector[i] ** 2 + yvector[i] ** 2),
                      df.columns[i]) for i in range(df.shape[1])],
                    reverse=True)
print(f"Projected features by importance:\n{importance}")
# Project original features onto your principal component feature-space:
for i in range(df.shape[1]):
    # Use an arrow to project each original feature:
    ax.arrow(0, 0, xvector[i], yvector[i],
             color='black', width=0.0005, head_width=0.02, alpha=0.75)
    ax.text(xvector[i] * 1.1, yvector[i] * 1.1, list(df.columns)[i],
            color='black', alpha=0.75)
# Add the cluster label back into the dataframe and display it:
df['label'] = pd.Series(labels, index=df.index)
print(df.head())
# Show graphs:
plt.show()
