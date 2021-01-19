# Classification-and-Clustering
### ENEE436 Project 2

The library Scikit-learn was used for implementing classification algorithms (SVMs and Neural Nets) and clustering algorithms (K-means and Spectral Clustering). Cross validation was used to get more representative test accuracies to determine optimal parameters for certain algorithms. Cross validation was implemented by combining the given training and testing data sets, using sklearn’s KFold method to randomly assign data points to a certain number of folds. After using each fold to train the model and using the remaining folds as the test set, the average test score would provide a more accurate metric for determining how well the model would behave. This was used to determine optimal parameters for both the SVM and Neural Net algorithms. The data sets provided were Banana, Twonorm, Waveform, two_spirals, crescent_and_the_full_moon, and cluster_within_cluster. The last three were provided as .mat files, which needed to be read using scipy instead of as raw data.

From the sklearn library, the SVC class was used for the Support Vector Machine Classification, the class MLPClassifier was used for the Neural Network classifier. The SpectralClustering class and the KMeans class were used for their respective clustering algorithms. The GaussianMixture class was used to implement the Gaussian mixture model algorithm. The KFolds class was used to create a k-folds cross-validation algorithm. The mlxtend library is a library built on matplotlib. It was used to draw the decision boundaries for the classification algorithms.

## Results
### KMeans Clustering
![K Means Clustering](https://github.com/prchandr/Classification-and-Clustering/blob/main/images/kMeansClustering.png?raw=true)

### Spectral Clustering
![Spectral Clustering](https://github.com/prchandr/Classification-and-Clustering/blob/main/images/spectralClustering.png?raw=true)

### Neural Net Boundaries
![Neural Net Boundary](https://github.com/prchandr/Classification-and-Clustering/blob/main/images/neuralNetBoundaries.png?raw=true)

### Support Vector Machine Boundaries
![SVM Boundary](https://github.com/prchandr/Classification-and-Clustering/blob/main/images/svmBoundaries.png?raw=true)

## Dependencies
In order to run these programs, make sure you have Numpy, Scipy, Scikit-learn, Matplotlib, and mlxtend installed. Numpy and Scikit-learn are used for the machine learning and classification. Matplotlib and mlxtend are used create plots and plot the decision boundaries. Scikit is used for importing the .mat data files.

Make sure mlxtend is installed before running svm.py or neuralNet.py. Ensure that the dataset files are in a folder called "data" in the same directory as the python program.

## How to run
Each program can be run through the command line with
> python <program_name>.py

