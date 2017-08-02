import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import math


def load(path_train, path_test):
    # Load up the data:
    testing = pd.read_csv(path_test, header=None)
    training = pd.read_csv(path_train, header=None)
    x_train = training.iloc[:, :-1]
    x_test = testing.iloc[:, :-1]
    y_train = training.iloc[:, -1]
    y_test = testing.iloc[:, -1]
    # Keep just the first 4% of the samples?
    four_p = False
    if four_p:
        x_train = x_train[:math.ceil(x_train.shape[0] * 0.04)]
        y_train = y_train[:math.ceil(y_train.shape[0] * 0.04)]
    return x_train, x_test, y_train, y_test


def peek_data(x_train):
    print("Peeking your data...")
    fig = plt.figure()
    fig.set_tight_layout(True)
    cnt = 1
    for col in range(5):
        for row in range(10):
            plt.subplot(5, 10, cnt)
            plt.imshow(x_train.iloc[cnt, :].values.reshape(8, 8),
                       cmap='gray_r', interpolation='nearest')
            plt.axis('off')
            cnt += 1


def draw_predictions(model, x_test, y_test):
    fig = plt.figure()
    fig.set_tight_layout(True)
    # Make some guesses
    y_guess = model.predict(x_test)
    # Do multi-plots:
    cols = 10
    rows = 5
    cnt = 1
    for col in range(cols):
        for row in range(rows):
            plt.subplot(rows, cols, cnt)
            # Plot the image in 8x8 pixels:
            plt.imshow(x_test.iloc[cnt, :].values.reshape(8, 8),
                       cmap='gray_r', interpolation='nearest')
            # Green -> Right / Red -> Fail:
            fontcolor = 'g' if y_test[cnt] == y_guess[cnt] else 'r'
            plt.title(f'Label: {y_guess[cnt]}',
                      fontsize=6, color=fontcolor)
            plt.axis('off')
            cnt += 1


# Pass in the file paths to the .tes and the .tra files:
X_train, X_test, Y_train, Y_test = load('datasets/optdigits.tra',
                                        'datasets/optdigits.tes')
# Get to know your data:
peek_data(X_train)
# Create an SVC classifier and train the model:
print("Training SVC Classifier...")
# Try different kernel: linear, poly, rbf.
svc = SVC(kernel='rbf', C=1, gamma=.001)
svc.fit(X_train, Y_train)
# Calculate the score of your SVC against the testing data:
print("Scoring SVC Classifier...")
score = svc.score(X_test, Y_test)
print(f"Score:\n{score}")
# Visual Confirmation of accuracy:
draw_predictions(svc, X_test, Y_test)
plt.savefig('plots/svc.png')
# Print out the TRUE value of the 1000th digit in the test set:
true_1000th_test = Y_test[999]
print(f"1000th test label: {true_1000th_test}")
# Predict the value of the 1000th digit in the test set:
guess_1000th_test = svc.predict(X_test.iloc[999, :].values.reshape(1, -1))[0]
print(f"1000th test prediction: {guess_1000th_test}")
# Display the 1000th test image:
fig = plt.figure()
plt.imshow(X_test.iloc[999, :].values.reshape(8, 8),
           cmap='gray_r', interpolation='nearest')

plt.show()
