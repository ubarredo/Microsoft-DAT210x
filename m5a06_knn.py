import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier


def plot_2d_boundary(model, d_train, l_train, d_test, l_test):
    fig, ax = plt.subplots()
    ax.set_title('Transformed Boundary, Image Space -> 2D')
    padding = 0.1
    resolution = 1
    # Calculate the boundaries of the mesh grid:
    ranges = np.ptp(d_train, axis=0)
    x_max, y_max = np.max(d_train, axis=0) + ranges * padding
    x_min, y_min = np.min(d_train, axis=0) - ranges * padding
    # Make the 2D Grid Matrix:
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Plot the mesh grid:
    ax.contourf(xx, yy, pred, alpha=.5, cmap='viridis')
    # Plot the test images:
    for n in range(d_test.shape[0]):
        # Set image shape:
        x0, y0 = d_test[n] - ranges * 0.025
        x1, y1 = d_test[n] + ranges * 0.025
        # Plot the image:
        img = pics[l_test.index[n], :].reshape(num_pixels, num_pixels)
        ax.imshow(img, aspect='auto', interpolation='nearest', zorder=100000,
                  extent=(x0, x1, y0, y1), cmap='gray', alpha=.8)
    # Plot the train data:
    ax.scatter(d_train[:, 0], d_train[:, 1], c=l_train, cmap='viridis',
               alpha=.8, marker='o', edgecolors='black')


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
# Load up your face_labels dataset as a series:
labels = pd.read_csv('datasets/face_labels.csv', header=None)[0]
# Do train_test_split:
X_train, X_test, Y_train, Y_test = train_test_split(pics, labels,
                                                    test_size=.15,
                                                    random_state=7)
# Implement Isomap:
iso = Isomap(n_components=2, n_neighbors=5)
iso.fit(X_train)
X_train = iso.transform(X_train)
X_test = iso.transform(X_test)  # Implement KNeighborsClassifier:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
# Print the accuracy of the testing set:
print(f"Accuracy: {knn.score(X_test, Y_test)}")
# Plot the decision boundary, the training data and testing images:
plot_2d_boundary(knn, X_train, Y_train, X_test, Y_test)
# Show graph:
plt.show()
