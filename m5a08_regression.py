import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def draw_line(model, data, label, title):
    # Plot the observations and display the R2 coefficient:
    fig, ax = plt.subplots()
    ax.scatter(data, label, c='g', marker='o')
    ax.plot(data, model.predict(data), color='orange', linewidth=1, alpha=.7)
    ax.set_title(title + ' R2: ' + str(model.score(data, label)))
    for i in [2014, 2030, 2045]:
        print(f"Est {i} {title} Life Expectancy: {model.predict([[i]])[0][0]}")


# Load up the data:
X = pd.read_csv('datasets/life_expectancy.csv', sep='\t', index_col=0)
print(X.head())
print(X.describe())
print(X.dtypes)
# Create linear regression model:
lreg = LinearRegression()
# Slice out your data manually:
X_train = pd.DataFrame(X.loc[:1986].index)
Y_train = pd.DataFrame(X.loc[:1986, 'WhiteMale'])
# Fit your model:
lreg.fit(X_train, Y_train)
# Pass it into draw_line with your training data:
draw_line(lreg, X_train, Y_train, 'WhiteMale')
# Print the actual 2014 WhiteMale life expectancy:
print(f"Act 2014 WhiteMale Life Expectancy: {X.loc[2014, 'WhiteMale']}")
# Repeat for BlackFemales:
Y_train = pd.DataFrame(X.loc[:1986, 'BlackFemale'])
lreg.fit(X_train, Y_train)
draw_line(lreg, X_train, Y_train, 'BlackFemale')
print(f"Act 2014 BlackFemale Life Expectancy: {X.loc[2014, 'BlackFemale']}")
# Display a correlation matrix for the entire dataset:
corre = X.corr()
print(corre)
plt.figure()
plt.imshow(corre, cmap='Reds', interpolation='nearest')
plt.colorbar()
tick_marks = range(len(X.columns))
plt.xticks(tick_marks, X.columns)
plt.yticks(tick_marks, X.columns)
# Show graphs:
plt.show()
