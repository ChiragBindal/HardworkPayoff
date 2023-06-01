#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:50:49 2023

@author: chiragbindal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Data Download
X = pd.read_csv('Linear_X_Train.csv')
y = pd.read_csv('Linear_Y_Train.csv')

X = X.values
y = y.values

#Visualisation
plt.style.use('seaborn')
plt.scatter(X, y)
plt.show()

# Normalisation 
u = np.mean(X)
std = np.std(X);
X = (X-u)/std

def hypothesis(x,theta) :
    y_ = theta[0] + theta[1]*x
    return y_

def gradient(X,Y,theta) :
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m) :
        y_ = hypothesis(X[i], theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y)*X[i]
    return grad/m

def error(X,Y,theta) :
    m = X.shape[0]
    total_error = 0.0
    for i in range(m) :
        y_ = hypothesis(X[i], theta)
        y = Y[i]
        total_error += (y_ - Y[i])**2
    return total_error/m


def gradientDescent(X,Y,max_steps=100,learning_rate=0.1):
    theta = np.zeros((2,))
    error_list  = []
    for i in range(max_steps) :
        grad = gradient(X, Y, theta)
        e = error(X,Y,theta)
        error_list.append(e)
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        
    return theta,error_list

theta,error_list = gradientDescent(X, y)
print(theta)
plt.plot(error_list)



