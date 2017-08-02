import os
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
# Plain python lists:
samples = []
colors = []
# Append each image to the list:
rpath = 'datasets/aloi/32r/'
for png in os.listdir(rpath):
    img = misc.imread(rpath + png)
    # Resample:
    img = img[::2, ::2]
    samples.append(img.flatten() / 255)
    colors.append('b')
ipath = 'datasets/aloi/32i/'
for png in os.listdir(ipath):
    img = misc.imread(ipath + png)
    # Resample:
    img = img[::2, ::2]
    samples.append(img.flatten() / 255)
    colors.append('r')
# Convert the list to a dataframe
df = pd.DataFrame(samples)
# Implement Isomap here. Reduce the dataframe df down to 3 components:
iso = manifold.Isomap(n_components=3, n_neighbors=6)
iso_samples = iso.fit_transform(df)
# Create a 2D Scatter plot:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(iso_samples[:, 0], iso_samples[:, 1], marker='o', c=colors)
# Create a 3D Scatter plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iso_samples[:, 0], iso_samples[:, 1], iso_samples[:, 2],
           marker='o', c=colors)
# Show graphs:
plt.show()
