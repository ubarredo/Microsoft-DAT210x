import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np

plt.style.use('ggplot')
# Load dataset and remove NaNs
df = pd.read_csv('datasets/kidney_disease.csv', index_col=0)
df.dropna(inplace=True)
# Create color coded labels:
labels = ['red' if i == 'ckd' else 'green' for i in df['classification']]
# Index 'bgr','wc' and 'rc' columns:
df = df[['bgr', 'wc', 'rc']]
# Check dtypes:
print(f"Types:\n{df.dtypes}")
df = df.apply(pd.to_numeric, errors='coerce')
print(f"New Types:\n{df.dtypes}")
# Check the variance and describe:
print(f"Variances:\n{df.var()}")
print(f"Describe:\n{df.describe()}")
# Scale features?
scale = True
# Standardize features by removing the mean and scaling to unit variance:
if scale:
    df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df),
                      columns=df.columns)
    print(f"New Variances:\n{df.var()}")
    print(f"New Describe:\n{df.describe()}")
# Run PCA reduce to 2 components:
pca = PCA(n_components=2)
t_df = pd.DataFrame(pca.fit_transform(df),
                    columns=['component1', 'component2'])
# Scale the principal components to estimate the variance:
xvector = pca.components_[0] * max(t_df.iloc[:, 0]) / 2  # type: np.ndarray
yvector = pca.components_[1] * max(t_df.iloc[:, 1]) / 2  # type: np.ndarray
# Show the importance of each feature:
importance = sorted([(math.sqrt(xvector[i] ** 2 + yvector[i] ** 2),
                      df.columns[i]) for i in range(df.shape[1])],
                    reverse=True)
print(f"Features by importance:\n{importance}")
# Project original features onto your principal component feature-space:
for i in range(df.shape[1]):
    # Use an arrow to project each original feature:
    plt.arrow(0, 0, xvector[i], yvector[i],
              color='b', width=0.0005, head_width=.1, alpha=0.75)
    plt.text(xvector[i] * 1.1, yvector[i] * 1.1, list(df.columns)[i],
             color='b', alpha=0.75)
ax = plt.axes()
# Plot the transformed data as a scatter plot using pandas:
t_df.plot.scatter(x='component1', y='component2',
                  marker='o', c=labels, alpha=0.75, ax=ax)
plt.savefig('plots/pca.png')
# Show the graph:
plt.show()
