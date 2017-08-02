import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates, andrews_curves
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
student_dataset = pd.read_csv('datasets/students.data',
                              index_col=0)

# HISTOGRAMS
my_series = student_dataset['G1']
my_series.plot.hist(alpha=.5)
my_dataframe = student_dataset[['G1', 'G2', 'G3']]
my_dataframe.plot.hist(alpha=.5)

# SCATTER 2D
my_dataframe.plot.scatter(x='G1', y='G3')

# SCATTER 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')
ax.scatter(student_dataset['G1'],
           student_dataset['G3'],
           student_dataset['Dalc'],
           c='r',
           marker='.')

# PARALLEL COORDINATES
data = load_iris()
df1 = pd.DataFrame(data['data'], columns=data['feature_names'])
df1['target_names'] = [data['target_names'][i] for i in data['target']]
plt.figure()
parallel_coordinates(df1, 'target_names')

# ANDREWS CURVES
plt.figure()
andrews_curves(df1, 'target_names')

# CORRELATION
plt.style.use('classic')
plt.figure()
df2 = pd.DataFrame(np.random.rand(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
plt.imshow(df2.corr(), cmap='Blues', interpolation='nearest')
plt.colorbar()
tick_marks = range(len(df2.columns))
plt.xticks(tick_marks, df2.columns, rotation='vertical')
plt.yticks(tick_marks, df2.columns)

plt.show()
