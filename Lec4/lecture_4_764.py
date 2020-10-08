# ####################################
# Group ID : 764
# Members : Bjørn Utrup Dideriksen, Kasper Steensig Jensen, Kristoffer Calundan Derosche
# Date : 2020/09/23
# Lecture: 4 Dimensionality Reduction
# Dependencies: numpy=1.19.2, scipy=1.5.2, matplotlib=3.3.2
# Python version: 3.8.2
# Functionality: Compute a 2D PCA of MNIST classes 5,6,8 and then classify them
# Example: 
# ###################################

import numpy as np
import numpy.linalg as la
from scipy import io
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

data = io.loadmat('mnist_all.mat')

X5, X6, X8 = data['train5'], data['train6'], data['train8']

# Create single data frame for PCA
X = np.concatenate((data['train5'], data['train6'], data['train8']))

mean_X = X.mean(axis=0)

# Covariance matrix estimate
S = (1 / (783)) * X.transpose().dot(X)
# Something funky is going on here, we are getting errors when using np.cov. Specifically look into S_e row 12 and 13.
# S1 = np.cov(X.transpose())
# S_e = S1-S

# V=eigvals,W=eigvectors
V, W = la.eig(S)

# sort the eigvals and eigvecs
idx = np.argsort(V)
V = V[idx]
W = W[:, idx]

# reduce to W to a 2D projection by selecting eigvecs for 2 largest eigvals
W = W[:, 0:2]

Y5 = np.matmul(X5, W)
Y6 = np.matmul(X6, W)
Y8 = np.matmul(X8, W)
Y = np.matmul(X, W)

Y5 = Y5 - Y5.mean(axis=0).copy()
Y6 = Y6 - Y6.mean(axis=0).copy()
Y8 = Y8 - Y8.mean(axis=0).copy()
Y = Y - Y.mean(axis=0).copy()

fig = plt.figure()
plt.scatter(Y5[:, 0], Y5[:, 1], color='r')
plt.scatter(Y6[:, 0], Y6[:, 1], color='g')
plt.scatter(Y8[:, 0], Y8[:, 1], color='b')
plt.show()


# Compute mean and covariance of Yn = Xn*W
def train_Yn(Xn):
    Yn = np.matmul(Xn, W)
    mu = Yn.mean(axis=0)
    var = np.cov(Yn.transpose())
    return mu, var


mu_Y5, var_Y5 = train_Yn(X5)
mu_Y6, var_Y6 = train_Yn(X6)
mu_Y8, var_Y8 = train_Yn(X8)

# Determine likelihood function of Yn, and likelihood of point x in Ln
Ln = lambda x, mu, var: multivariate_normal.pdf(x, mu, var)


def test(data, classification, scale):
    correct = 0
    for x in data:
        c = classification
        y = x.dot(W)  # transform point x into point y
        y = y - Y.mean(axis=0)

        # Determine posteriors of y being 5,6,8
        l_Y5 = scale[0] * Ln(y, mu_Y5, var_Y5)
        l_Y6 = scale[1] * Ln(y, mu_Y6, var_Y6)
        l_Y8 = scale[2] * Ln(y, mu_Y8, var_Y8)

        # Use the specified test class c to determine correct classfication
        if l_Y5 > l_Y6 and l_Y5 > l_Y8 and c == 5:
            correct = correct + 1
        elif l_Y6 > l_Y5 and l_Y6 > l_Y8 and c == 6:
            correct = correct + 1
        elif l_Y8 > l_Y5 and l_Y8 > l_Y6 and c == 8:
            correct = correct + 1

    return correct, len(data)


correct_Y5, len_Y5 = test(data['test5'], 5, (1 / 3, 1 / 3, 1 / 3))
correct_Y6, len_Y6 = test(data['test6'], 6, (1 / 3, 1 / 3, 1 / 3))
correct_Y8, len_Y8 = test(data['test8'], 8, (1 / 3, 1 / 3, 1 / 3))

correct_tot = correct_Y5 + correct_Y6 + correct_Y8
len_tot = len_Y5 + len_Y6 + len_Y8

print(correct_tot / len_tot)
