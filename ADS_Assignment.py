import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('green_house_gas_Emission.csv')
df.head()
df.info()
print(df.describe())
print(df.describe().T)


centres = [[-1., 0.], [1., -0.5], [0., 1.]]

import sklearn.datasets as skdat
import numpy as np
import matplotlib.pyplot as plt

xy, nclust = skdat.make_blobs(1000, centers=centres, cluster_std=0.3)

x = xy[:,0] # extract x and y vectors
y = xy[:,1]

centres_arr = np.array(centres)  # Convert centres to a numpy array
xcent = centres_arr[:, 0]
ycent = centres_arr[:, 1]

print(xcent)
print(ycent)


# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(x, y, 10, nclust, marker="o", cmap=cm)
plt.scatter(xcent, ycent, 45, "k", marker="d")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet
ncluster = 3

kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(xy) 
labels = kmeans.labels_

cen = kmeans.cluster_centers_
print(cen)

print(skmet.silhouette_score(xy, labels))

plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster):
    plt.plot(x[labels==l], y[labels==l], "o", markersize=4, color=col[l])

for ic in range(ncluster):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print(kmeans.predict([[0.5, 0.5]]))

ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
# calculate the silhoutte score
print(skmet.silhouette_score(xy, labels))



centres = [[-1., 0.], [1., -0.5], [0., 1.]]
import sklearn.datasets as skdat
import numpy as np
import matplotlib.pyplot as plt

xy, nclust = skdat.make_blobs(1000, centers=centres, cluster_std=0.3)


for i in range(20):
  print(xy[i], nclust[i])
  
  
  
  
x = xy[:,0] # extract x and y vectors
y = xy[:,1]

centres_arr = np.array(centres)  # Convert centres to a numpy array
xcent = centres_arr[:, 0]
ycent = centres_arr[:, 1]

print(xcent)
print(ycent)

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(x, y, 10, nclust, marker="o", cmap=cm)
plt.scatter(xcent, ycent, 45, "k", marker="d")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet
ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy) # fit done on x,y pairs
labels = kmeans.labels_
# print(labels) # labels is the number of the associated clusters of (x,y)␣
# for i in range(50):
# print(xy[i], labels[i])
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)
# calculate the silhoutte score
print(skmet.silhouette_score(xy, labels))
# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:yellow", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster): # loop over the different labels
    plt.plot(x[labels==l], y[labels==l], "o", markersize=4, color=col[l])
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()



print(kmeans.predict([[0.5, 0.5]]))



ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
# calculate the silhoutte score
print(skmet.silhouette_score(xy, labels))


print(df.describe())


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X= df.iloc[: , 1: :]
X




df[X.columns] = scaler.fit_transform(df[X.columns])

df_green = df[['1990', '2000', '2010', '2015', '2019']]
print(df_green.describe())

import cluster_tools as ct
corr = df_green.corr()
print(corr)


plt.figure(figsize=(15, 10))
corr = corr.fillna(0)
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix of Green House Gas Emission")
plt.show()



pd.plotting.scatter_matrix(df_green, figsize=(12, 12), s=5, alpha=0.8)
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_function(x, a, b):
    """Linear function for fitting.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The input x-values.
    a : float
        The slope of the line.
    b : float
        The y-intercept of the line.
        
    Returns:
    --------
    numpy.ndarray
        The predicted y-values for the given x-values.
    """
    return a * x + b

# Load data from CSV file
data = np.genfromtxt('green_house_gas_Emission.csv', delimiter=',', skip_header=1, usecols=(1,6))

# Remove NaN and infinite values
mask = np.logical_and(np.isfinite(data[:, 0]), np.isfinite(data[:, 1]))
data = data[mask]

x = data[:, 0]  # First column as x-vector
y = data[:, 1]  # Second column as y-vector

# Fit linear regression using curve_fit
popt, pcov = curve_fit(linear_function, x, y)

# Extract fitted parameters
a_fit, b_fit = popt

# Create predicted y-values using fitted parameters
y_pred = linear_function(x, a_fit, b_fit)

# Plot the original data and the fitted line
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, 'r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the fitted parameters
print("Fitted Parameters:")
print("a =", a_fit)
print("b =", b_fit)