import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
# Load 'wheat.data':
seeds = pd.read_csv('datasets/wheat.data', index_col=0)
# Create 'area', 'perimeter' and 'assymetry' 3d scatter plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(seeds['area'],
           seeds['perimeter'],
           seeds['asymmetry'],
           c='red',
           marker='.')
# Create 'width', 'groove' and 'length' 3d scatter plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('width')
ax.set_ylabel('groove')
ax.set_zlabel('length')
ax.scatter(seeds['width'],
           seeds['groove'],
           seeds['length'],
           c='green',
           marker='.')
# Display the graphs:
plt.show()
