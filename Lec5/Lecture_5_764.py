# ####################################
# Group ID : 764
# Members : Bjørn Utrup Dideriksen, Kasper Steensig Jensen, Kristoffer Calundan Derosche
# Date : 2020/09/30
# Lecture: 5 Clustering
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
from scipy.cluster.vq import kmeans

data = io.loadmat('data/2D568class.mat')

trn5, trn6, trn8 = data['trn5_2dim'], data['trn6_2dim'], data['trn8_2dim']

trn_combined = np.concatenate((trn5,trn6,trn8))

fig = plt.figure()
plt.scatter(trn5[:, 0], trn5[:, 1], color='r')
plt.scatter(trn6[:, 0], trn6[:, 1], color='g')
plt.scatter(trn8[:, 0], trn8[:, 1], color='b')
plt.show()

fig = plt.figure()
plt.scatter(trn_combined[:, 0], trn_combined[:, 1], color='r')
plt.show()

means = kmeans(trn_combined, 3)[0]
covs = 600**2*np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]])

L0 = lambda x: multivariate_normal.pdf(x, means[0], covs[0])
L1 = lambda x: multivariate_normal.pdf(x, means[1], covs[1])
L2 = lambda x: multivariate_normal.pdf(x, means[2], covs[2])

L = [L0,L1,L2]
classes = len(L)

Priors = np.full((classes), 1/classes)

def Estep(X, means, covs, Priors):
    Z = np.zeros((len(X),classes))
    for j, xt in enumerate(X):
        l = list(map(lambda Li: Li(xt), L))
        l = np.array(l)*Priors
        
        posteriors=l
  
        i = np.argmax(posteriors)      
        Z[j, i] = 1
    
    return Z

def Mstep(X, Z, means):
    for i in range(classes):
        index_vec = Z[:, i]
        index_vec_ = np.array([Z[:, i]])
        Priors[i] = sum(index_vec)/len(Z)
        means[i] = sum(X*index_vec_.transpose())/sum(index_vec)
        
        covs[i] = np.zeros((2,2))
        
        for j, xt in enumerate(X):
            inner_product = np.outer((xt-means[i]).transpose(), xt-means[i])
            covs[i] = covs[i] + index_vec[j]*inner_product
            
        covs[i] = covs[i]/sum(index_vec)
    return means, covs, Priors

print(means)
print(covs)

print('\n')

for i in range(2):
    Z = Estep(trn_combined, means, covs, Priors)
    means, covs, Priors = Mstep(trn_combined, Z, means)
    
    #print(multivariate_normal.pdf(np.array([-250, -500]), means[0], covs[0]))
    
#print(Priors)
print(means)
print(covs)

Z0 = Estep(data['tst5_2dim'], means, covs, Priors)
Z1 = Estep(data['tst6_2dim'], means, covs, Priors)
Z2 = Estep(data['tst8_2dim'], means, covs, Priors)

print(sum(Z0[:,0])/len(Z0), sum(Z0[:,1])/len(Z0), sum(Z0[:,2])/len(Z0))
print(sum(Z1[:,0])/len(Z1), sum(Z1[:,1])/len(Z1), sum(Z1[:,2])/len(Z1))
print(sum(Z2[:,0])/len(Z2), sum(Z2[:,1])/len(Z2), sum(Z2[:,2])/len(Z2))

data_x = np.linspace(-1500, 2000, 200)
data_y = np.linspace(-1500, 1000, 200)
X, Y = np.meshgrid(data_x, data_y)
z0 = np.zeros([200,200])
z1 = np.zeros([200,200])
z2 = np.zeros([200,200])

for i,x in enumerate(data_x):
    for j,y in enumerate(data_y):
        z0[i,j] = L0(np.array([x,y]))
        z1[i,j] = L1(np.array([x,y]))
        z2[i,j] = L2(np.array([x,y]))

fig, ax = plt.subplots()
ax.scatter(trn5[:, 0], trn5[:, 1], color='r')
ax.scatter(trn6[:, 0], trn6[:, 1], color='g')
ax.scatter(trn8[:, 0], trn8[:, 1], color='b')
CS = ax.contour(X, Y, z0)
CS = ax.contour(X, Y, z1)
CS = ax.contour(X, Y, z2)

