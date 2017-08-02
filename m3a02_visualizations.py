import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Load 'wheat.data':
seeds = pd.read_csv('datasets/wheat.data', index_col=0)
# Create 'area' and 'perimeter' 2d scatter plot:
seeds.plot.scatter(x='area', y='perimeter', marker='^')
# Create 'groove' and 'asymmetry' 2d scatter plot:
seeds.plot.scatter(x='groove', y='asymmetry', marker='.')
# Create 'compactness' and 'width' 2d scatter plot:
seeds.plot.scatter(x='compactness', y='width', marker='o')
plt.savefig('plots/scatter.png')
# Display the graphs:
plt.show()
