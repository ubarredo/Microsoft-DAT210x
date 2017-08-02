import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load up the dataset and take a peek at its head and dtypes:
df = pd.read_csv('datasets/call_detail_records.csv',
                 dtype={'In': str, 'Out': str})
print(df.head())
print(f"Old dtypes:\n{df.dtypes}")
# Convert the date and the time:
df['CallDate'] = pd.to_datetime(df['CallDate'], errors='coerce')
df['Duration'] = pd.to_timedelta(df['Duration'], errors='coerce')
print(f"New dtypes:\n{df.dtypes}")
# Distinct list of "In" phone numbers:
users = df['In'].drop_duplicates().tolist()
# Run K-Means:
model = KMeans(n_clusters=2)
# Color space:
colors = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(users))).tolist()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for u in users:
    # Create a slice that filters to selected user:
    usern = df[df['In'] == u]
    colorn = colors.pop()
    print(f"User {u}")
    # Print out its length:
    print(f"- Total Calls: {len(usern)}")
    # Plot all the call locations:
    ax1.scatter(usern['TowerLon'], usern['TowerLat'], c=colorn,
                marker='o', edgecolor='black', alpha=.8, label=u)
    # Filter examining only weekends before 6am or after 10pm:
    usern = usern[((usern['DOW'] == 'Sat') |
                   (usern['DOW'] == 'Sun')) &
                  ((usern['CallTime'] < '06:00:00') |
                   (usern['CallTime'] > '22:00:00'))]
    print(f"- Weekend Calls (<6am or >10pm): {len(usern)}")
    ax2.scatter(usern['TowerLon'], usern['TowerLat'], c=colorn,
                marker='o', edgecolor='black', alpha=.8, label=u)
    model.fit(usern[['TowerLon', 'TowerLat']])
    centroids = model.cluster_centers_
    print(f"- Centroids:\n{centroids}\n")
    ax2.scatter(centroids[:, 0], centroids[:, 1], c=colorn,
                marker='x', linewidths=3, s=169)
# Show graphs:
ax1.set_title('Total Calls')
ax1.legend()
ax2.set_title('Weekend Calls (<6am or >10pm)')
ax2.legend()
plt.show()
