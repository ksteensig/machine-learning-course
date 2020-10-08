# ####################################
# Group ID : 764
# Members : Bjørn Utrup Dideriksen, Kasper Steensig Jensen, Kristoffer Calundan Derosche
# Date : 2020/10/07
# Lecture: 6 Linear Discriminant Analysis
# Dependencies: numpy=1.19.2, scipy=1.5.2, matplotlib=3.3.2
# Python version: 3.8.2
# Functionality: Compute a 2D PCA of MNIST classes 5,6,8 and then classify them
# Example: 
# ###################################

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

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

train = np.concatenate(np.array(train, dtype=object))
test = np.concatenate(np.array(test, dtype=object))
labels = np.concatenate(np.array(labels, dtype=object)).transpose()
test_labels = np.concatenate(np.array(test_labels, dtype=object)).transpose()

# LDA
lda = LinearDiscriminantAnalysis(n_components = 2).fit(train, labels)
lda_2d = lda.transform(train)
lda_2d_test = lda.transform(test)

fig = plt.figure()
for i in range(10):
    plt.scatter(lda_2d[labels == i, 0], lda_2d[labels == i, 1])
plt.show()

gnb_lda = GaussianNB().fit(lda_2d, labels)
pred_lda = gnb_lda.predict(lda_2d_test)

lda_accuracy = (pred_lda != test_labels).sum()/len(test_labels)

# PCA
pca = PCA(n_components=2).fit(train)
pca_2d = pca.transform(train)
pca_2d_test = pca.transform(test)

fig = plt.figure()
for i in range(10):
    plt.scatter(pca_2d[labels == i, 0], pca_2d[labels == i, 1])
plt.show()

gnb_pca = GaussianNB().fit(pca_2d, labels)
pred_pca = gnb_pca.predict(pca_2d_test)

pca_accuracy = (pred_pca != test_labels).sum()/len(test_labels)