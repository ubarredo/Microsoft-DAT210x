import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def benchmark(model, d_train, d_test, l_train, l_test, title):
    print(f"\n{title} Results")
    s = time.time()
    for i in range(iterations):
        # Train the classifier on the training data:
        model.fit(d_train, l_train)
    print(f"{iterations} Iterations Training Time: {time.time() - s}")
    s = time.time()
    for i in range(iterations):
        # Score the classifier on the testing data:
        score = model.score(d_test, l_test)
    print(f"{iterations} Iterations Scoring Time: {time.time() - s}")
    print(f"High-Dimensionality Score: {round(score*100, 3)}")


def draw_plots(model, d_train, d_test, l_train, l_test, title):
    """ Function to break any higher-dimensional space down and view cross
    sections of it."""
    plt.style.use('ggplot')
    # Parameters:
    padding = 1
    resolution = 10
    max_2d_score = 0
    n = d_train.shape[1]
    fig, axes = plt.subplots(n, n)
    fig.canvas.set_window_title(title)
    fig.set_tight_layout(True)
    plt.setp(axes, xticks=(), yticks=())
    for row, col in [(a, b) for a in range(n) for b in range(n)]:
        # Intersection:
        if col == row:
            axes[row, col].text(0.5, 0.5, d_train.columns[row],
                                verticalalignment='center',
                                horizontalalignment='center',
                                fontsize=12)
            continue
        # Select two features to display:
        d_train_bag = d_train.iloc[:, [row, col]]
        d_test_bag = d_test.iloc[:, [row, col]]
        # Create a mesh to plot in:
        x_max, y_max = d_train_bag.max() + padding
        x_min, y_min = d_train_bag.min() - padding
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        axes[row, col].set_xlim(x_min, x_max)
        axes[row, col].set_ylim(y_min, y_max)
        # Choose model:
        if row < col:
            colors = 'viridis'
            model.fit(d_train_bag, l_train)
            pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(
                xx.shape)
            axes[row, col].contourf(xx, yy, pred, cmap=colors, alpha=.5)
            axes[row, col].scatter(d_train_bag.iloc[:, 0],
                                   d_train_bag.iloc[:, 1],
                                   c=l_train, cmap=colors, alpha=.8)
            score = round(model.score(d_test_bag, l_test) * 100, 3)
            axes[row, col].text(0.5, 0, f'Score: {score}',
                                transform=axes[row, col].transAxes,
                                horizontalalignment='center',
                                fontsize=8)
            max_2d_score = score if score > max_2d_score else max_2d_score
    print(f"Max 2D Score: {max_2d_score}")


iterations = 5000
# Load up the wheat dataset into X:
X = pd.read_csv('datasets/wheat.data', index_col=0)
print(X.head())
# Which rows have NaNs?
print(X[X.isnull().any(axis=1)])
# Go ahead and drop any row with a NaN:
X.dropna(inplace=True)
# Copy the labels out of the dset into variable Y:
Y = X['wheat_type'].copy()
# Encode the labels using map:
Y = Y.map({'canadian': 0, 'kama': 1, 'rosa': 2})
# Remove labels from X:
X.drop('wheat_type', axis=1, inplace=True)
# Split your data into test / train sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.3,
                                                    random_state=7)
# Create an Decision Tree classifier:
dtr = DecisionTreeClassifier(max_depth=9, random_state=2)
# Do Decision Tree:
benchmark(dtr, X_train, X_test, Y_train, Y_test, 'DecisionTree')
# Draw:
draw_plots(dtr, X_train, X_test, Y_train, Y_test, 'DecisionTree')
# Show graphs:
plt.show()
