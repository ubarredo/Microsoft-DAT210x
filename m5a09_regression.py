import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


def draw_line(model, data, label, title, r2):
    # Plot the observations and display the R2 coefficient:
    fig, ax = plt.subplots()
    ax.scatter(data, label, c='g', marker='o')
    ax.plot(data, model.predict(data), color='orange', linewidth=1, alpha=.7)
    ax.set_title(title + ' R2: ' + str(r2))
    print(title + ' R2: ' + str(r2))
    print(f"Intercept(s): {model.intercept_}")


def draw_plane(model, data, label, title, r2):
    # Plot the observations and display the R2 coefficient:
    fig = plt.figure()
    ax = Axes3D(fig)
    # Set up a grid:
    x_max, y_max = np.max(data, axis=0)
    x_min, y_min = np.min(data, axis=0)
    x, y = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 10),
                       np.arange(y_min, y_max, (y_max - y_min) / 10))
    # Predict based on possible input values:
    z = model.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], label, c='g', marker='o')
    ax.plot_wireframe(x, y, z, color='orange', alpha=.7)
    ax.set_zlabel('prediction')
    ax.set_title(title + ' R2: ' + str(r2))
    print(title + ' R2: ' + str(r2))
    print(f"Intercept(s): {model.intercept_}")


plt.style.use('ggplot')
# Load up the dataset:
df = pd.read_csv('datasets/college.csv', index_col=0)
# Encode features directly:
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})
print(df.head())
# Create the linear regression model:
lreg = LinearRegression()
Y = df['Accept'].copy()
for i in ['Room.Board', 'Enroll', 'F.Undergrad']:
    # Index the slice:
    X = df[[i]].copy()
    # Use train_test_split to cut data:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=.3,
                                                        random_state=7)
    # Fit and score the model:
    lreg.fit(X_train, Y_train)
    score = lreg.score(X_test, Y_test)
    # Pass it into draw_line with your test data:
    draw_line(lreg, X_test, Y_test, f'Accept({i})', score)
plt.savefig('plots/regression.png')
# Do multivariate linear regression:
X = df[['Room.Board', 'Enroll']].copy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=.3,
                                                    random_state=7)
lreg.fit(X_train, Y_train)
score = lreg.score(X_test, Y_test)
# Pass it into draw_plane with your test data:
draw_plane(lreg, X_test, Y_test, "Accept(Room.Board,Enroll)", score)
# Show graphs:
plt.show()
