import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

# Make array of the different kernel coefficients to test
kernelCoefficients = np.ones((8,1))
for i in range(8):
	kernelCoefficients[i] = 1/np.power(2, i+1)

def kFoldTest(data, labels, CLFlist, kf, printEachVal=False):
	train_means = []
	test_means = []
	train_scores = []
	test_scores = [] 

	# Iterate through each classifier
	for clf in CLFlist:
		train_score = []
		test_score = []

		# Perform k-fold testing
		for train_index, test_index in kf.split(data):
			# Get new train/test data and labels
			train_data = data[train_index]
			test_data = data[test_index]
			train_labels = labels[train_index]
			test_labels = labels[test_index]

			# Train model
			clf.fit(train_data, train_labels)

			# Score model
			train_score.append(clf.score(train_data, train_labels))
			test_score.append(clf.score(test_data, test_labels))
		
		# Calculate means and store
		train_means.append(np.mean(train_score))
		train_scores.append(train_score)
		test_means.append(np.mean(test_score))
		test_scores.append(test_score)
		
		# Print each clf's values, if wanted
		if printEachVal: 
			print("\tKernel: ",  clf.kernel, " \t\tCoeff: ", clf.gamma)
			print("\tTrain Score: ", np.round(np.mean(train_score), 4), "\tTest Score: ",  np.round(np.mean(test_score),4))
			print("\tSupport vectors: ", clf.n_support_, "\n")

	# Get optimal SVM and coefficient
	optimal_idx = np.argmax(test_means)
	optimal_train_acc = train_means[optimal_idx]
	optimal_test_acc = test_means[optimal_idx]
	optimal_coef = 'linear' if optimal_idx == len(kernelCoefficients) else kernelCoefficients[optimal_idx]
	optimal_CLF = CLFlist[optimal_idx]

	# Print optimal info
	print("Optimal Coef: ", optimal_coef)
	print("Train Accuracy: ", optimal_train_acc)
	print("Test Accuracy: ", optimal_test_acc)
	print("Num of Support Vectors: ", optimal_CLF.n_support_)

	return optimal_CLF

def main():
	# load data
	wf_train_labels = np.loadtxt("data/waveform_train_labels_1.asc")
	wf_train_data = np.loadtxt("data/waveform_train_data_1.asc")
	wf_test_labels = np.loadtxt("data/waveform_test_labels_1.asc")
	wf_test_data = np.loadtxt("data/waveform_test_data_1.asc")

	tn_train_labels = np.loadtxt("data/twonorm_train_labels_1.asc")
	tn_train_data = np.loadtxt("data/twonorm_train_data_1.asc")
	tn_test_labels = np.loadtxt("data/twonorm_test_labels_1.asc")
	tn_test_data = np.loadtxt("data/twonorm_test_data_1.asc")

	bn_train_labels = np.loadtxt("data/banana_train_labels_1.asc")
	bn_train_data = np.loadtxt("data/banana_train_data_1.asc")
	bn_test_labels = np.loadtxt("data/banana_test_labels_1.asc")
	bn_test_data = np.loadtxt("data/banana_test_data_1.asc")

	# Merge training and testing data, separating this is unnecessary for cross-validation
	wf_labels = np.append(wf_train_labels, wf_test_labels, axis=0)
	wf_data = np.append(wf_train_data, wf_test_data, axis=0)
	tn_labels = np.append(tn_train_labels, tn_test_labels, axis=0)
	tn_data = np.append(tn_train_data, tn_test_data, axis=0)
	bn_labels = np.append(bn_train_labels, bn_test_labels, axis=0)
	bn_data = np.append(bn_train_data, bn_test_data, axis=0)

	# Make list of SVM Classifiers for each dataset. Linear is the last index.
	wf_linear_CLF = SVC(kernel='linear')
	tn_linear_CLF = SVC(kernel='linear')
	bn_linear_CLF = SVC(kernel='linear')

	wf_CLF = [SVC(kernel='rbf', gamma=c) for c in kernelCoefficients]
	tn_CLF = [SVC(kernel='rbf', gamma=c) for c in kernelCoefficients]
	bn_CLF = [SVC(kernel='rbf', gamma=c) for c in kernelCoefficients]

	wf_CLF.append(wf_linear_CLF)
	tn_CLF.append(tn_linear_CLF)
	bn_CLF.append(bn_linear_CLF)

	# 5-fold Cross Validation to determine optimal SVM
	kf = KFold(n_splits=5, shuffle=True)

	print("\nWaveform: ")
	kFoldTest(wf_data, wf_labels, wf_CLF, kf)

	print("\nTwo Norm: ")
	kFoldTest(tn_data, tn_labels, tn_CLF, kf)

	print("\nBanana: ")
	optimal_bn_CLF = kFoldTest(bn_data, bn_labels, bn_CLF, kf)

	# Plot training examples of Banana with boundary
	plot_decision_regions(bn_data, bn_labels.astype(np.integer), clf=optimal_bn_CLF, legend=2)
	plt.title("Decision boundary of Banana")
	plt.show()

	# PCA Waveform to 2D
	pca = PCA(n_components=2)
	wf_reduced_data = pca.fit_transform(wf_data)

	print("\nWaveform PCA-2D: ")
	optimal_wf_CLF = kFoldTest(wf_reduced_data, wf_labels, wf_CLF, kf)

	# Plot training examples of PCD-2D Waveform with boundary
	plot_decision_regions(wf_reduced_data, wf_labels.astype(np.integer), clf=optimal_wf_CLF, legend=2)
	plt.title("Decision boundary for PCA-2D of Waveform")
	plt.show()

if __name__ == '__main__':
	main()