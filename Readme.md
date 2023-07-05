# K-Means Clustering on Penguins Dataset

### Emil.io 2023

## This repository contains code for implementing K-Means clustering on the penguins dataset. The penguins dataset is a collection of measurements of various features of different penguin species, such as body mass, bill depth, and flipper length. The goal of the K-Means clustering algorithm is to group similar penguin species together into distinct clusters based on their measured features.

### Getting Started

To get started with the code, you will need to have Python 3 and some required packages installed on your computer. You can install the required packages using the following command:

pip install -r requirements.txt

### Dataset

The penguins dataset used in this repository is the "Palmer Penguins" dataset, which is a relatively small dataset consisting of 344 penguin observations with 8 variables.

The dataset can be downloaded from the following link: https://github.com/allisonhorst/palmerpenguins

### Code

The code for performing K-Means clustering on the penguins dataset is located in the "kmeans_penguins.py" file. This file contains the following functions:

    init_centroids: Initializes the K-Means centroids using random data points.
    assign_clusters: Assigns each data point to the closest centroid.
    update_centroids: Calculates the new centroids based on the mean of the data points in each cluster.
    kmeans: Performs the K-Means clustering algorithm on the data.
    calculate_tss: Calculates the total sum of squares for a given set of clusters and centroids.
    elbow_method: Uses the elbow method to determine the optimal number of clusters for the data.

### Usage

To use the code, simply run the "kmeans_penguins.py" file. The file contains a main function that loads the penguins dataset, performs K-Means clustering on the data, and outputs the resulting clusters and centroids.

To run the elbow method and determine the optimal number of clusters for the data, call the "elbow_method" function with the desired range of cluster values.
Conclusion

K-Means clustering is a powerful algorithm for grouping similar data points into distinct clusters. By applying this algorithm to the penguins dataset, we can gain insights into the relationships between different penguin species based on their measured features.
