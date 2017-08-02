import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D


def plot2d(data, t_data, title, x, y, faces=40):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(f'Component: {x}')
    ax.set_ylabel(f'Component: {y}')
    # Size of the images displayed:
    x_size = (max(t_data[:, x]) - min(t_data[:, x])) * 0.08
    y_size = (max(t_data[:, y]) - min(t_data[:, y])) * 0.08
    for n in range(faces):
        num = np.random.choice(range(data.shape[0]))
        # Image limit points:
        x0, x1 = t_data[num, x] - x_size / 2., t_data[num, x] + x_size / 2.
        y0, y1 = t_data[num, y] - y_size / 2., t_data[num, y] + y_size / 2.
        img = data[num, :].reshape(num_pixels, num_pixels)
        # Plot the image:
        ax.imshow(img, aspect='auto', cmap='gray', interpolation='nearest',
                  zorder=100000, extent=(x0, x1, y0, y1))
    # Plots the full scatter:
    ax.scatter(t_data[:, x], t_data[:, y], marker='.', alpha=.7)


def plot3d(t_data, title, x=0, y=1, z=2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel(f'Component: {x}')
    ax.set_ylabel(f'Component: {y}')
    ax.set_zlabel(f'Component: {z}')
    ax.scatter(t_data[:, x], t_data[:, y], t_data[:, z], marker='.', alpha=.7)


# Load the .mat file:
mat = scipy.io.loadmat('datasets/face_data.mat')
# Get the img data:
pics = mat['images'].transpose()
num_images = pics.shape[0]
num_pixels = int(np.sqrt(pics.shape[1]))
# Transpose the pictures:
for i in range(num_images):
    pics[i, :] = pics[i, :].reshape(num_pixels,
                                    num_pixels).transpose().flatten()
# Implement PCA. Reduce the dimensions to 3 components:
pca = PCA(n_components=3)
pca_pics = pca.fit_transform(pics)
# Implement Isomap. Reduce the dimensions to 3 components:
iso = Isomap(n_components=3, n_neighbors=8)
iso_pics = iso.fit_transform(pics)
# Call plot2d using the first 2 components:
plot2d(pics, pca_pics, 'PCA 2D', x=0, y=1)
plot2d(pics, iso_pics, 'ISO 2D', x=0, y=1)
plt.savefig('plots/isomap.png')
# Call plot3d:
plot3d(pca_pics, 'PCA 3D')
plot3d(iso_pics, 'ISO 3D')
# Show graphs:
plt.show()
