import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Load up the mushroom dataset:
headers = ['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?',
           'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
           'gill-color', 'stalk-shape', 'stalk-root',
           'stalk-surface-above-ring', 'stalk-surface-below-ring',
           'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
           'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
           'population', 'habitat']
X = pd.read_csv('datasets/agaricus_lepiota.data', names=headers)
# Check nan encoding:
X.replace('?', np.nan, inplace=True)
print(f"NaN values: \n{X.isnull().sum()}")
# Drop any row with a nan:
X.dropna(inplace=True)
print(f"New shape: {X.shape}")
# Copy the labels out of the dset and then remove it:
Y = X['label'].copy()
X.drop('label', axis=1, inplace=True)
# Encode the labels:
Y = Y.map({'p': 0, 'e': 1})
# Encode the entire dataset using dummies:
X = pd.get_dummies(X)
# Split the data into test / train sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.3,
                                                    random_state=7)
# Create a default DecisionTree classifier:
dtr = DecisionTreeClassifier()
# Train the classifier on the training data / labels:
dtr.fit(X_train, Y_train)
# Top features:
top_features = X_train.columns[np.argsort(dtr.feature_importances_)[::-1]]
print(f"Top two features: {top_features[:2].values}")
# Score the classifier on the testing data / labels:
score = dtr.score(X_test, Y_test)
print(f"High-Dimensionality Score: {round((score*100), 3)}")
# Output a .dot file:
export_graphviz(dtr.tree_, out_file='datasets/tree.dot',
                feature_names=X_train.columns)
# Then render the .dot to .png:
# http://webgraphviz.com/
