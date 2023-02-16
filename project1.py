'''
Created on Feb 9, 2023
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
# Edited slighlty by Emil.io, January 2023
import numpy as np
import pandas as pd
from math import sqrt
import random
"""
# loading the dataset
penguins = pd.read_csv("penguins.csv")
print(penguins)
penguins.dropna()

# This function should work for you
# Document what it is doing specifically with inline comments
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))[0,0]    

"""

penguins = pd.read_csv("penguins.csv")
print(penguins)
penguins.dropna()

def euclidean_distance(vecA, vecB):
    """Calculate the Euclidean distance between two points"""
    return np.sqrt(np.sum((vecA - vecB) ** 2))
x = euclidean_distance(penguins["BodyMass_g"],["Delta15N"]) 
print(x)

def init_centroids(penguins, k):
    """Initialize the centroids randomly from the data"""
    n_samples, n_features = penguins.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = penguins[random.randint(0, n_samples - 1), :]
        centroids[i, :] = centroid
    return centroids

def assign_clusters(penguins, centroids):
    """Assign each sample to the closest centroid"""
    n_samples = penguins.shape[0]
    clusters = np.zeros(n_samples)
    for i in range(n_samples):
        sample = penguins[i, :]
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters

def update_centroids(penguins, clusters, k):
    """Update the centroids based on the mean of the samples assigned to each cluster"""
    n_samples, n_features = penguins.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        cluster_samples = penguins[clusters == i, :]
        centroid = np.mean(cluster_samples, axis=0)
        centroids[i, :] = centroid
    return centroids

def kmeans(penguins, k, max_iterations=100):
    """Perform K-Means Clustering on the data"""
    centroids = init_centroids(penguins, k)
    for i in range(max_iterations):
        clusters = assign_clusters(penguins, centroids)
        new_centroids = update_centroids(penguins, clusters, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Example usage
penguins = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
k = 2
centroids, clusters = kmeans(penguins, k)
print("Centroids:", centroids)
print("Clusters:", clusters)


import matplotlib
import matplotlib.pyplot as plt
def showPlt(penguins, alg=kmeans, numClust=5):
    myCentroids, clustAssing = alg(penguins, numClust)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = kmeans[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
