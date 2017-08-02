import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Load 'wheat.data':
seeds = pd.read_csv('datasets/wheat.data', index_col=0)
# Drop 'area' and 'perimeter' features:
seeds.drop(['area', 'perimeter'], axis=1, inplace=True)
# Plot a parallel coordinates chart grouped by 'wheat_type':
plt.figure()
parallel_coordinates(seeds, 'wheat_type', alpha=.4)
plt.savefig('plots/parallel_coordinates.png')
# Display the graph:
plt.show()
