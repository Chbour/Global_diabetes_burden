from numpy.linalg import norm
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

def kmeans(data, k, maxIterations, distance="cosinus", vectorColumn="text_vec"):
    random.seed(0)
    # Initialize centroids randomly
    centroids = initialise_centroids(data, k)
    
    return kmeans_loop(data, centroids, k, maxIterations, distance)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    # return data

def kmeans_loop(data, centroids, k, maxIterations, distance):
    # Initialize book keeping vars.
    oldCentroids = np.array([np.zeros((300,)) for i in range(k)])
    
    # Run the main k-means algorithm
    for iteration in tqdm(range(maxIterations)):
        if shouldStop(oldCentroids, centroids):
            break
        
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        
        # Assign labels to each datapoint based on centroids
        labels = getLabelForAllDataPoints_new(data, centroids, distance)

        # Assign centroids based on datapoint labels
        centroids = getNewCentroids(data, labels.astype(int), k)
    
    return labels


def getLabelForAllDataPoints_new(data, centroids, distance):
    mat_prod = np.matmul(data,centroids.transpose())
    nV = norm(data, axis=1)
    nV = nV.reshape(nV.shape[0], 1)
    nC = norm(centroids, axis=1)

    mat_mult=np.multiply(nV,nC)
    R = np.divide(mat_prod,mat_mult)
    return np.argmax(R, axis=1)
    

def getNewCentroids(data, labels, k):
    vect_sum = np.zeros((k, data.shape[1]))
    count_label = np.zeros((k,))
    for i in range(labels.shape[0]):
        label = labels[i]
        vect = data[i,:]
        vect_sum[label,:] += vect
        count_label[label] += 1
    
    for i in range(k):
        vect_sum[i,:] = vect_sum[i,:]/count_label[i]
    return vect_sum


def initialise_centroids(data, k):
    random_indices = random.sample(range(data.shape[0]), k)
    
    return data[random_indices,:]


def shouldStop(oldCentroids, centroids):
    return np.array_equal(centroids, oldCentroids)
