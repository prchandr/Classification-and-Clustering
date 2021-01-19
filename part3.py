import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from spectral import spectralCluster
from kmeans import kMeansCluster

# load data
wf_train_labels = (np.loadtxt("data/waveform_train_labels_1.asc") * 0.5 + 0.5).astype(int)			# Change from -1 and 1's to 0 and 1
wf_train_data = np.loadtxt("data/waveform_train_data_1.asc")

# Get clustering models
kmeans = kMeansCluster(wf_train_data, numClusters=2, drawPlot=False)
sc = spectralCluster(wf_train_data, numClusters=2, drawPlot=False)

# Make sure classes are consistent. Force first value to be 1, which is the value of the actual label
kmeansLabels = kmeans.labels_
if kmeansLabels[0] == 0:
	kmeansLabels = (~kmeansLabels.astype(bool)).astype(int)

scLabels = sc.labels_
if scLabels[0] == 0:
	scLabels = (~scLabels.astype(bool)).astype(int)

# Get error rates
kMeansError = 1- np.sum(kmeansLabels==wf_train_labels) / len(kmeansLabels)
scError = 1 - np.sum(scLabels==wf_train_labels) / len(scLabels)

print("KMeans Error rate: " + str(kMeansError))
print("Spectral Error rate: " + str(scError))