
# coding: utf-8

# In[148]:

from matplotlib.pyplot import plot, show
import numpy as np
from numpy.random import rand, random, uniform
from sklearn import cluster


# In[133]:

# Constants
FIRST_CIRCLE_RADIUS = 10
FIRST_CIRCLE_STD = 2
FIRST_CIRCLE_NUM_POINT = 100

SECOND_CIRCLE_RADIUS = 20
SECOND_CIRCLE_STD = 2
SECOND_CIRCLE_NUM_POINT = 200


# In[134]:

# Get circles

def get_circle(radius, radius_std, num_point):
    circle = np.array([]).reshape(0, 2)
    for i in range(num_point):
        distance_to_origin = radius + random()*radius_std*2 - 2
        angle = uniform(low=0.0, high=np.pi*2)
        x_value = np.cos(angle) * distance_to_origin
        y_value = np.sin(angle) * distance_to_origin
        point = np.array([x_value, y_value])
        circle = np.concatenate((circle, [point]))
    return circle
    
first_circle = get_circle(FIRST_CIRCLE_RADIUS, FIRST_CIRCLE_STD, FIRST_CIRCLE_NUM_POINT)
second_circle = get_circle(SECOND_CIRCLE_RADIUS, SECOND_CIRCLE_STD, SECOND_CIRCLE_NUM_POINT)


# In[140]:

# Test plot

for point in first_circle:
    plot(point[0], point[1], 'go')

for point in second_circle:
    plot(point[0], point[1], 'go')

show()


# In[142]:

data = np.concatenate((first_circle, second_circle))


# In[144]:

# K-means
# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()


# In[156]:

# Spectral clustering

spectral = cluster.SpectralClustering(n_clusters=2,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")

spectral.fit(data)

y_preds = spectral.labels_.astype(np.int)

for index, y_pred in enumerate(y_preds):
    if y_pred == 0:
        plot_option = 'ob'
    else:
        plot_option = 'or'
    
    plot(data[index][0], data[index][1], plot_option)

show()

