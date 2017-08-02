import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def plot_decision_boundary(model, data, labels):
    fig, ax = plt.subplots()
    padding = 0.1
    resolution = 100
    # Calculate the boundaris:
    ranges = np.ptp(data, axis=0)
    x_max, y_max = np.max(data, axis=0) + ranges * padding
    x_min, y_min = np.min(data, axis=0) - ranges * padding
    # Create a 2D Grid Matrix with the predictions of the class:
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Plot the contour map:
    ax.contourf(xx, yy, pred, alpha=.5, cmap='viridis')
    # Plot the original points:
    ax.scatter(data[:, 0], data[:, 1],
               c=labels, cmap='viridis', edgecolors='black', )
    ax.set_title(f"K = {model.get_params()['n_neighbors']}")


# Load in the dataset:
X = pd.read_csv('datasets/breast_cancer_wisconsin.data',
                header=None,
                index_col=0,
                names=['sample', 'thickness', 'size', 'shape',
                       'adhesion', 'epithelial', 'nuclei', 'chromatin',
                       'nucleoli', 'mitoses', 'status'])
print(f"Old types\n{X.dtypes}")
X = X.apply(pd.to_numeric, errors='coerce')
print(f"New types\n{X.dtypes}")
# Copy out the 'status' column into a slice:
Y = X['status'].copy()
# Drop 'status' from the main:
X.drop('status', axis=1, inplace=True)
# Fill each row's NaNs with the mean of the feature:
X.fillna(X.mean(), inplace=True)
# Do train_test_split:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.5,
                                                    random_state=7)
# Choose data preprocessing scaler:
choice = 3
if choice == 1:
    proc = preprocessing.StandardScaler()
elif choice == 2:
    proc = preprocessing.Normalizer()
elif choice == 3:
    proc = preprocessing.MaxAbsScaler()
elif choice == 4:
    proc = preprocessing.MinMaxScaler()
elif choice == 5:
    proc = preprocessing.RobustScaler()
else:
    proc = False
if proc:
    proc.fit(X_train)
    X_train = proc.transform(X_train)
    X_test = proc.transform(X_test)
# PCA or Isomap?
test_pca = True
if test_pca:
    # Implement PCA:
    print("Computing 2D Principle Components")
    redu = PCA(n_components=2)
else:
    # Implement Isomap:
    print("Computing 2D Isomap Manifold")
    redu = Isomap(n_components=2, n_neighbors=7)
# Train model against train data:
redu.fit(X_train)
# Transform train and test data:
X_train = redu.transform(X_train)
X_test = redu.transform(X_test)
# Implement and train KNeighborsClassifier:
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train, Y_train)
# Show the accuracy of the testing set:
print(f"Accuracy: {knn.score(X_test, Y_test)}")
plot_decision_boundary(knn, X_test, Y_test)
# Show graph:
plt.show()
