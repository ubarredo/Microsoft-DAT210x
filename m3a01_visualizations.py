import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Load 'wheat.data':
seeds = pd.read_csv('datasets/wheat.data', index_col=0)
# Slice 'area' and 'perimeter':
s1 = seeds[['area', 'perimeter']]
# Slice 'groove' and 'asymmetry':
s2 = seeds[['groove', 'asymmetry']]
# Create histograms:
s1.plot.hist(alpha=.75)
plt.savefig('plots/histogram.png')
s2.plot.hist(alpha=.75)
# Display the graphs:
plt.show()
