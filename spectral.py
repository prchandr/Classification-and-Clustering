import numpy as np
from sklearn.cluster import SpectralClustering
import scipy.io
import matplotlib.pyplot as plt

def spectralCluster(data, numClusters, drawPlot=True, title=''):
	if title == "":
		title = "SpectralClustering with " + str(numClusters) + " clusters"

	# Fit a Spectral Clustering Model to the data.
	sc = SpectralClustering(n_clusters=numClusters,gamma=0.1,affinity='nearest_neighbors',n_jobs=-1)
	sc.fit(data)

	# Plot clusters
	if drawPlot:
		plt.title(title)
		plt.scatter(data[:,0], data[:,1], c=sc.labels_)
		plt.show()

	return sc

def main():
	num_clusters = [2, 4]

	# Load data
	two_spirals = scipy.io.loadmat('data/two_spirals.mat')['data']
	crescent_and_the_full_moon = scipy.io.loadmat('data/crescent_and_the_full_moon.mat')['data']
	cluster_within_cluster = scipy.io.loadmat('data/cluster_within_cluster.mat')['data']

	for n in num_clusters:
		for dataset in [two_spirals, crescent_and_the_full_moon, cluster_within_cluster]:
			spectralCluster(dataset, n)


if __name__ == '__main__':
	main()
