import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 

df = pd.read_csv('customers.csv')
print(df.head(10))

#all 
print(df.all())

# tail
print(df.tail())

# shape 
print(df.shape)

# info 

print(df.info())

x = df.iloc[:, [3,4]].values
y = df.iloc[:, [3,4]].values[0:20]
print(x)

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS values')
plt.show()
