import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Load up the data set:
X = pd.read_csv('datasets/parkinsons.data')
# Drop the 'name' column:
X.drop('name', axis=1, inplace=True)
# Splice out the status column:
Y = X['status'].copy()
X.drop('status', axis=1, inplace=True)
# Perform a train/test split:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.3,
                                                    random_state=7)
# Program a best-parameter search:
scalers = {'NoScaler': False,
           'StandardScaler': preprocessing.StandardScaler(),
           'Normalizer': preprocessing.Normalizer(),
           'MaxAbsScaler': preprocessing.MaxAbsScaler(),
           'MinMaxScaler': preprocessing.MinMaxScaler(),
           'RobustScaler': preprocessing.RobustScaler()}
best_score = 0
for sk, sv in scalers.items():
    proc = sv
    if proc:
        proc.fit(X_train)
        tX_train = proc.transform(X_train)
        tX_test = proc.transform(X_test)
    else:
        tX_train = X_train.copy()
        tX_test = X_test.copy()
    # Check dimensionality reduction? (PCA, Isomap, None)
    choice = 2
    if choice == 1:
        n1 = 4
        redu = PCA(n_components=n1)
    elif choice == 2:
        n1 = 4
        n2 = 3
        redu = Isomap(n_components=n1, n_neighbors=n2)
    else:
        redu = False
    if redu:
        redu.fit(tX_train)
        ttX_train = redu.transform(tX_train)
        ttX_test = redu.transform(tX_test)
    else:
        ttX_train = tX_train.copy()
        ttX_test = tX_test.copy()
    for c, g in [(a, b) for a in np.arange(0.05, 2.05, 0.05)
                 for b in np.arange(0.001, 0.101, 0.001)]:
        svc = SVC(C=c, gamma=g)
        svc.fit(ttX_train, Y_train)
        score = svc.score(ttX_test, Y_test)
        if score > best_score:
            best_score = score
            best_scaler = sk
            best_C = c
            best_gamma = g
print("Optimal combination:")
print(f"\tScore: {best_score}")
print(f"\tScaler: {best_scaler}")
print(f"\tC: {best_C}")
print(f"\tgamma: {best_gamma}")
