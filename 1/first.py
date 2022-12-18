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

#  instigating Elbow method to get numbers of all optimal clusters
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

# Using unsupervised learning algorithm to train model i.e. KMEANS

k_means_model = KMeans(n_clusters=5, init = 'k-means++', random_state=0)
y_kmeans = k_means_model.fit_predict(x)
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans==0,1], s = 80, c = 'red', label = 'Customer 1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans==1,1], s = 80, c = 'blue', label = 'Customer 2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans==2,1], s = 80, c = 'green', label = 'Customer 3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans==3,1], s = 80, c = 'yellow', label = 'Customer 4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans==4,1], s = 80, c = 'purple', label = 'Customer 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],s =100, c= 'magenta', label = 'Centroids')
plt.title("Cluster of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()

