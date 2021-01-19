import numpy as np
from sklearn.cluster import KMeans
import scipy.io
import matplotlib.pyplot as plt

def kMeansCluster(data, numClusters, drawPlot=True, title=""):
	if title == "":
		title = "KMeansCluster with " + str(numClusters) + " clusters"

	# Fit a KMeans clustering model to the data
	kmeans = KMeans(n_clusters=numClusters).fit(data)

	if drawPlot:
		plt.title(title)
		plt.scatter(data[:,0], data[:,1], c=kmeans.labels_)
		plt.show()

	return kmeans

def main():
	num_clusters = [2, 4]

	# Load data
	two_spirals = scipy.io.loadmat('data/two_spirals.mat')['data']
	crescent_and_the_full_moon = scipy.io.loadmat('data/crescent_and_the_full_moon.mat')['data']
	cluster_within_cluster = scipy.io.loadmat('data/cluster_within_cluster.mat')['data']

	# Perform the cluster on each dataset for each number of clusters
	for n in num_clusters:
		for dataset in [two_spirals, crescent_and_the_full_moon, cluster_within_cluster]:
			kMeansCluster(dataset, n)


if __name__ == '__main__':
	main()