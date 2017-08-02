import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def plot_decision_boundary(model, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
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
    plt.contourf(xx, yy, pred, alpha=.5, cmap='viridis')
    # Plot the original points:
    plt.scatter(data[:, 0], data[:, 1],
                c=labels, cmap='viridis', edgecolors='black', )
    plt.title(f"K = {model.get_params()['n_neighbors']}")


plt.style.use('ggplot')
# Load up the dataset into a variable called X and print its head:
X = pd.read_csv('datasets/wheat.data', index_col=0)
print(X.head())
# Copy the 'wheat_type' series into a series called Y:
Y = X['wheat_type'].copy()
# Drop the original 'wheat_type' column from the X:
X.drop('wheat_type', axis=1, inplace=True)
# Do a "nominal" categorization of Y:
Y = Y.astype('category').cat.codes
# Fill each row's NaNs with the mean of the feature:
X.fillna(X.mean(), inplace=True)
# Split X and Y into training and testing data:.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.33,
                                                    random_state=1)
# Create an instance of SKLearn's Normalizer class and train it:
norm = preprocessing.Normalizer()
norm.fit(X_train)
# Transform training and testing data:
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
# Create a PCA transformation and Fit it against training data:
pca = PCA(n_components=2)
pca.fit(X_train)
# Project training and testing features into PCA space:
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# Create and train a KNeighborsClassifier:
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_pca, Y_train)
# Plot decision boundary:
plot_decision_boundary(knn, X_train_pca, Y_train)
plt.savefig('plots/knn.png')
# Display the accuracy score of your test data/labels:
print(f"Accuracy: {knn.score(X_test_pca, Y_test)}")
# Show graph:
plt.show()
