# ####################################
# Group ID : 764
# Members : Bjørn Utrup Dideriksen, Kasper Steensig Jensen, Kristoffer Calundan Derosche
# Date : 2020/09/16
# Lecture: 3 Parametric Methods
# Dependencies: numpy=1.19.1, scipy=1.5.2, matplotlib=3.3.2, pandas=1.1.1
# Python version: 3.8.2
# Functionality: Short Description.
# Example: 
# ###################################

from numpy import std, mean, cov
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt

trn_x=pd.read_csv('dataset1_G_noisy_ASCII/trn_x.txt',sep='\s+',header=None)
trn_y=pd.read_csv('dataset1_G_noisy_ASCII/trn_y.txt',sep='\s+',header=None)

mean_x = trn_x.mean()
var_x = trn_x.cov()

print(mean_x)
print(var_x)

mean_y = trn_y.mean()
var_y = trn_y.cov()

trn_x = trn_x.to_numpy()

Lx = lambda x: multivariate_normal.pdf(x, mean_x, var_x)
Ly = lambda y: multivariate_normal.pdf(y, mean_y, var_y)

def test(data, classes, scale):
    correct = 0
    
    for index, point in enumerate(data):
        c = int(classes[index])
        lx = scale[0]*Lx(point)
        ly = scale[1]*Ly(point)
        
        if lx > ly and c == 1:
            correct = correct+1
        elif ly >= lx and c == 2:
            correct = correct+1
            
    return correct, len(data)

tst_xy=pd.read_csv('dataset1_G_noisy_ASCII/tst_xy.txt',sep='\s+',header=None).to_numpy()
tst_xy_class=pd.read_csv('dataset1_G_noisy_ASCII/tst_xy_class.txt',sep='\s+',header=None).to_numpy()

a_correct, a_guesses = test(tst_xy, tst_xy_class, (0.5,0.5))
print(a_correct/a_guesses)

tst_xy_126=pd.read_csv('dataset1_G_noisy_ASCII/tst_xy_126.txt',sep='\s+',header=None).to_numpy()
tst_xy_126_class=pd.read_csv('dataset1_G_noisy_ASCII/tst_xy_126_class.txt',sep='\s+',header=None).to_numpy()

b_correct, b_guesses = test(tst_xy_126, tst_xy_126_class, (0.5,0.5))
print(b_correct/b_guesses)

c_correct, c_guesses = test(tst_xy_126, tst_xy_126_class, (0.9,0.1))
print(c_correct/c_guesses)