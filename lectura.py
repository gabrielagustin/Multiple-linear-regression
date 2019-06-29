# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 24 10:16:04 2018

@author: gag 

"""

import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# import statistics


from sklearn.preprocessing import normalize
from sklearn import preprocessing

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def normalizado(c):
    min = np.min(c)
    max = np.max(c)
    new = (c -min)/(max-min)
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new


def lecturaCompleta_etapa3(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    #print (data)
    print ("Numero inicial de muestras: " + str(len(data)))
    #data.SM_CONAE = data.SM_CONAE *100
    data.SM_SMAP = data.SM_SMAP *100
#    fig, ax = plt.subplots()
#    sns.distplot(data.PP)
    data.PP = data.PP * 0.1
    
#    fig, ax = plt.subplots()
#    sns.distplot(data.T_s)
    data.T_s = data.T_s -273.15
#    fig, ax = plt.subplots()
#    sns.distplot(data.T_s_modis)
    dataNew = data[(data.T_s_modis > 250)]
    data = dataNew
    
    data.T_s_modis = data.T_s_modis -273.15

    fig, ax = plt.subplots()
    sns.distplot(data.T_s_modis)
#    
#    
#    
#    fig, ax = plt.subplots()
#    sns.distplot(data.Et)
    dataNew = data[(data.Et < 3000)]
    data = dataNew
#    fig, ax = plt.subplots()
#    sns.distplot(data.Et)
    
    data.Et = ((data.Et*0.1)/8)/0.035
#    fig, ax = plt.subplots()
#    sns.distplot(data.Et)
    # plt.show()
#    


    ## se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    # print("percentile back 5: " + str(perc5Back))
    perc90Back = math.ceil(np.percentile(data.Sigma0, 99))
    # print("percentile back 95: " + str(perc90Back))
    dataNew = data[(data.Sigma0 > -18) & (data.Sigma0 < -4)]
    data = dataNew   

    ### se filtra el rango de valores de HR
#    perc5HR = math.ceil(np.percentile(data.HR,0))
##    print "percentile HR 5: " + str(perc5HR)
#    perc90HR = math.ceil(np.percentile(data.HR, 99))
#    print "percentile HR 95: " + str(perc90HR)
#    dataNew = data[(data.HR > 17.83) & (data.HR < 83.63)]
#    data = dataNew


    #print "Numero de muestras: " + str(len(data))

#    # se filtra el rango de valores de RSOILTEMPC
#    perc5Ts = math.ceil(np.percentile(data.T_s,0))
##    print ("percentile Ts 5: " + str(perc5Ts))
#    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
##    print ("percentile Ts 95: " + str(perc90Ts))
#    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < 25)]
#    data = dataNew
    #print ("Numero de muestras: " + str(len(data)))


    #se filtra el rango de valores de evapotransporacion
    #perc10Et = math.ceil(np.percentile(data.Et, 5))
    #print ("percentile Et 5: " + str(perc10Et))
    #perc90Et = math.ceil(np.percentile(data.Et, 75))
    #print ("percentile Et 90: " + str(perc90Et))
    #print ("Filtro por Et")
#    dataNew = data[(data.Et > 50) & (data.Et <= 450)]
#    data = dataNew
#    dataNew = data[(data.Et > 550) ]
#    data = dataNew
    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.1) & (data.NDVI < 0.8)]
    data = dataNew
    # print ("Filtro por NDVI")
    # print ("Numero de muestras: " + str(len(data)))

    del data['NDVI']
    del data['Date']
#    del data['T_s_modis']
    

    print("Numero de muestras luego de ser filtradas: " + str(len(data))) 



    print("-----------------------------------------------------------------------")
    print ("Estadisticos de las variables filtradas sin normalizar"+'\n')
    print(data.describe())

#    print('---MLR---------------------------------------------------------------')
#    print ("Variables sin normalizar")
#    print(data.describe())
#    print('------------------------------------------------------------------')

    data.PP = normalizado(data.PP)
    data.Sigma0 = normalizado(data.Sigma0)
    data.T_s = normalizado(data.T_s)
    data.Et = normalizado(data.Et)
    data.T_s_modis = normalizado(data.T_s_modis)


#    print('------------------------------------------------------------------')
    data.SM_SMAP = np.log10(data.SM_SMAP)
    data.T_s = np.log10(data.T_s)
    data.T_s_modis = np.log10(data.T_s_modis)
    data.Et = np.log10(data.Et)
    
#    del data['T_s_modis']
#    del data['T_s']
#    print('------------------------------------------------------------------')
#    print(data.describe())
#    print('------------------------------------------------------------------')
    return data
