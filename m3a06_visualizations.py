import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Load 'wheat.data':
seeds = pd.read_csv('datasets/wheat.data', index_col=0)
# Compute the correlation matrix:
correlations = seeds.drop('wheat_type', axis=1).corr()
# Graph the correlation matrix:
plt.figure()
plt.imshow(correlations, interpolation='nearest')
plt.colorbar()
tick_marks = list(range(len(correlations)))
plt.xticks(tick_marks, correlations.columns, rotation='vertical')
plt.yticks(tick_marks, correlations.columns)
plt.savefig('plots/correlations.png')
# Display the graph:
plt.show()
