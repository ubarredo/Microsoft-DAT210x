import pandas as pd
import datetime as dt
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def do_normal_pca(df):
    # Import the libraries required for PCA:
    from sklearn.decomposition import PCA
    # Reduce to 2D:
    normal_pca = PCA(n_components=2)
    # Train PCA on the armadillo:
    normal_pca.fit(df)
    # Project the armadillo:
    return normal_pca.transform(df)


def do_random_pca(df):
    # Import the libraries required for PCA:
    from sklearn.decomposition import PCA
    # Reduce to 2D randomized:
    random_pca = PCA(n_components=2, svd_solver='randomized')
    # Train PCA on the armadillo and Project to 2D:
    return random_pca.fit_transform(df)


plt.style.use('ggplot')
reduce_factor = 100
# Load up the scanned armadillo:
ply = PlyData.read('datasets/stanford_armadillo.ply')
armadillo = pd.DataFrame({'x': ply['vertex']['z'][::reduce_factor],
                          'y': ply['vertex']['x'][::reduce_factor],
                          'z': ply['vertex']['y'][::reduce_factor]})
# Render the original armadillo:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Armadillo 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(armadillo['x'], armadillo['y'], armadillo['z'],
           c='green', marker='.', alpha=0.75)
# Time the execution of nPCA 5000x:
t = dt.datetime.now()
npca = do_normal_pca(armadillo)
for i in range(4999):
    do_normal_pca(armadillo)
time_delta = dt.datetime.now() - t
# Render the transformed PCA armadillo:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('NormalPCA, build time: ' + str(time_delta))
ax.scatter(npca[:, 0], npca[:, 1], c='blue', marker='.', alpha=0.75)
# Time the execution of rPCA 5000x
t = dt.datetime.now()
rpca = do_random_pca(armadillo)
for j in range(4999):
    do_random_pca(armadillo)
time_delta = dt.datetime.now() - t
# Render the transformed RandomizedPCA armadillo:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('RandomizedPCA, build time: ' + str(time_delta))
ax.scatter(rpca[:, 0], rpca[:, 1], c='red', marker='.', alpha=0.75)
# Display the graphs:
plt.show()
