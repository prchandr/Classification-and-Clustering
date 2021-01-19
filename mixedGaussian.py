import numpy as np
import scipy.io
from sklearn.mixture import GaussianMixture

def gaussianMix(data):
	gm = GaussianMixture(n_components=2)
	gm.fit(data)

	print("Means:")
	print(gm.means_)

	print("Covariances: ")
	print(gm.covariances_)

def main():
	# Load data
	two_spirals = scipy.io.loadmat('data/two_spirals.mat')['data']
	crescent_and_the_full_moon = scipy.io.loadmat('data/crescent_and_the_full_moon.mat')['data']
	cluster_within_cluster = scipy.io.loadmat('data/cluster_within_cluster.mat')['data']
	
	# Get the gaussian mixtures of 2 components for each dataset
	print("Two Spirals: ")
	gaussianMix(two_spirals)

	print("Crescent and Full Moon: ")
	gaussianMix(crescent_and_the_full_moon)

	print("Cluster Within Cluster: ")
	gaussianMix(cluster_within_cluster)

if __name__ == '__main__':
	main()