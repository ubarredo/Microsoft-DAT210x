import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load up the dataset:
X = pd.read_csv('datasets/human_activity.csv', delimiter=';', decimal=',')
# Encode the gender column:
X['gender'] = X['gender'].map({'Man': 0, 'Woman': 1})
# Check data types
print(X.dtypes)
# Drop any problematic records:
X.drop(122076, inplace=True)
# Convert any column that needs to be converted into numeric:
X['z4'] = pd.to_numeric(X['z4'], errors='raise')
# Encode your labels value as a dummies version of 'class':
Y = pd.get_dummies(X['class'])
# TGet rid of the 'user' and 'class' columns:
X.drop(['user', 'class'], axis=1, inplace=True)
# Describe features:
print(X.describe())
# Check NaNs:
print(f"NaNs:\n{X.isnull().sum()}")
# Create an RandomForest classifier:
rfc = RandomForestClassifier(n_estimators=30, max_depth=10, oob_score=True,
                             random_state=0)
# Split data into test / train sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3,
                                                    random_state=7)
print("Fitting...")
s1 = time.time()
# Train your model on your training set:
rfc.fit(X_train, Y_train)
print(f"Fitting completed in: {time.time() - s1}")
# Display the OOB Score of the data:
score1 = rfc.oob_score_
print(f"OOB Score: {round(score1*100, 3)}")
print("Scoring...")
s2 = time.time()
# Score your model on your test set:
score2 = rfc.score(X_test, Y_test)
print("Score: ", round(score2 * 100, 3))
print(f"Scoring completed in: {time.time() - s2}")
