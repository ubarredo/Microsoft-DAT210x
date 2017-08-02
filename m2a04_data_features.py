import pandas as pd

# Load up the table, and extract the dataset:
url = 'http://www.espn.com/' \
      'nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
df = pd.read_html(url, header=1)[0]
print(df)
print(df.dtypes)
# Rename some columns:
df = df.rename(columns={'G.1': 'PP_G',
                        'A.1': 'PP_A',
                        'G.2': 'SH_G',
                        'A.2': 'SH_A'})
# Remove rows that have at least 4 NANs:
df = df.dropna(axis=0, thresh=4)
# Select all rows except the erroneous ones:
df = df[df['RK'] != 'RK']
# Get rid of the 'RK' column:
df = df.drop('RK', 1)
# Reset the index and don't store the original index
df = df.reset_index(drop=True)
# Correct the data type of all columns:
df[df.columns[2:]] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
print(df)
print(df.dtypes)
# Number of rows:
print(len(df))
# Number of unique PCT values:
print(len(df['PCT'].unique()))
# Value you get by adding the GP values at indices 15 and 16:
print(sum(df.loc[15:16, 'GP']))
