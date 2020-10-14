# ####################################
# Group ID : 764
# Members : Bjørn Utrup Dideriksen, Kasper Steensig Jensen, Kristoffer Calundan Derosche
# Date : 2020/10/14
# Lecture: 7 Support Vector Machines
# Dependencies: numpy=1.19.2, scipy=1.5.2, matplotlib=3.3.2, scikit-learn=0.23.2
# Python version: 3.8.2
# Functionality: Classify MNIST dataset using an SVM
# ###################################

import numpy as np
from scipy import io
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = io.loadmat('data/mnist_all.mat')

train = []
labels = []
test = []
test_labels = []

for i in range(10):
    train.append(data['train' + str(i)])
    
    label = np.zeros((len(data['train' + str(i)])))
    label.fill(i)
    labels.append(label)
    
    test.append(data['test' + str(i)])
    
    test_label = np.zeros((len(data['test' + str(i)])))
    test_label.fill(i)
    test_labels.append(test_label)

# concatenate training data/labels and test data/labels into their own numpy arrays
train = np.concatenate(np.array(train, dtype=object))
test = np.concatenate(np.array(test, dtype=object))
labels = np.concatenate(np.array(labels, dtype=object)).transpose()
test_labels = np.concatenate(np.array(test_labels, dtype=object)).transpose()

# use LDA to reduce the MNIST data to 9 dimensions, can also use PCA with 10 dimensions


def lda_svm():
    lda = LinearDiscriminantAnalysis(n_components = 9).fit(train, labels)
    lda_train = lda.transform(train)
    lda_test = lda.transform(test)

    svm_mnist = svm.SVC(verbose=True).fit(lda_train, labels)

    predictions = svm_mnist.predict(lda_test)
    correct = sum(test_labels == predictions)
    
    return correct/len(test_labels)

print('9D LDA transformed SVM accuracy: ', lda_svm())

def full_svm():
    svm_mnist = svm.SVC(verbose=True).fit(train, labels)

    predictions = svm_mnist.predict(test)
    correct = sum(test_labels == predictions)
    
    return correct/len(test_labels)

print('Full SVM accuracy: ', full_svm())