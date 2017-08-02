import pandas as pd

# Load up the dataset and describe the features:
df = pd.read_csv('datasets/tutorial.csv')
print(df.describe())
# Index with: [2:4,'col3'] and print the results:
print(df.loc[2:4, 'col3'])
