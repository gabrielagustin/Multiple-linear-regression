# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018
@author: gag 
"""

#Librerias
import numpy as np
from numpy import linalg as LA
from io import StringIO
from scipy.linalg import *
from sklearn import datasets, linear_model
import RLM
import statsmodels.formula.api as smf
#import scikits.statsmodels.api as sm

import sklearn
from sklearn.metrics import mean_squared_error


def calc_vif(matrix):
    numRows, numCols = matrix.shape
    vif_list = np.zeros((numCols,1))
    #print vif_list
    for j in range (0,numCols):
        y = np.zeros((numRows,1))
        for g in range(0,numRows):
            y[g] = matrix[g,j]
        #print y.shape
        #print y
        data = remove(matrix,j)
        #print data
        #print "----------------------------------------------------------------"
        coeff = RLM.training(data, y)
        #print coeff
        outVector = RLM.test(coeff, data)
        R2 = deterCoeff(y, outVector)
        vif = 1. / (1. - R2)
        vif_list[j] = vif
    return vif_list


def deterCoeff(y,yAprox):
    '''
    Compute determination coefficient
    '''
    mean = np.mean(y)
    SCT=0
    SCR=0
    ### variabilidad del modelo
    for i in range (0,len(y)):
        SCT = SCT + (y[i]-mean) * (y[i]-mean)
    ### variabilidad explicada
    for i in range (0,len(y)):
        SCR = SCR + ((y[i]-yAprox[i]) * (y[i]-yAprox[i]))
    R2 = 1-(SCR/SCT)
    #R2 = sklearn.metrics.r2_score(y, yAprox)
    #print "Coeficiente de determinación multiple Ṛ squared:" + str(R2)
    return R2


def RMSE(y,yAprox):
    '''
    Compute root-mean-square error
    '''
    error = 0
    e = 0
    num = len(y)
    print (num)
    for i in range(0,num):
        e = e + (y[i]-yAprox[i])**2
    error = np.sqrt(e/num)
    return error


def MSE(y,yAprox):
    '''
    Compute root-mean-square error
    '''
    mse = mean_squared_error(y, yAprox)
    return mse


def bias(yEstimated, yMeasured):
    N = len(yEstimated)
    bias = np.sum(yEstimated - yMeasured)/N
    return bias