import pandas as pd

# Load up the dataset with correct header column labels:
df = pd.read_csv('datasets/servo.data',
                 header=None,
                 names=['motor', 'screw', 'pgain', 'vgain', 'class'])
# Number of samples that have vgain equal to 5:
print(len(df[df['vgain'] == 5]))
# Number of samples that have motor and screw equal to E:
print(len(df[(df['motor'] == 'E') & (df['screw'] == 'E')]))
# Mean vgain value for the samples with pgain equal to 4:
print(df[df['pgain'] == 4]['vgain'].mean())
# See data types:
print(df.dtypes)
