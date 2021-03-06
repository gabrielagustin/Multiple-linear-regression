# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018
@author: gag 
"""

import numpy as np
import statistics
import lectura
import pandas as pd
import copy

import selection_methods


file = "/home/gag/MyProjects/Multiple-linear-regression/DataSet_9km.csv"
data = lectura.lecturaCompleta_etapa3(file)


#fileCal = 'tabla_completa_Calibracion_dataSet_1km.csv'
#fileVal = 'tabla_completa_Validacion_dataSet_1km.csv'
#
#dataCal = lectura.lecturaCompleta_etapa3(fileCal)
#dataVal = lectura.lecturaCompleta_etapa3(fileVal)
#
#data = [dataCal, dataVal]
#data = pd.concat(data)



#print(data)


# se mezclan las observaciones de las tablas
# semilla para mezclar los datos en forma aleatoria
np.random.seed(1)
dataNew = selection_methods.shuffle(data)
dataNew = data.reset_index(drop=True)
dataNew = data


model1 = selection_methods.forward_selected(data, 'SM_SMAP')
formula = model1.model.formula
print("---------------------------------------------------------------------")
print("Modelo planteado with forward selection and Rsquare: " + str(formula))
print("R^2 del modelo: " + str(model1.rsquared))
print("---------------------------------------------------------------------")



# model2 = selection_methods.forward_selected_vif(data, 'SM_SMAP')
# formula = model2.model.formula
# print("---------------------------------------------------------------------")
# print("Modelo planteado with forward selection and VIF: " + str(formula))
# print("R^2 del modelo: " + str(model2.rsquared))
# print("---------------------------------------------------------------------")




###  the stepwise method configured to function as backward elimination
model3 = selection_methods.stepwise_selection(data, 'SM_SMAP') ##['Et', 'PP', 'Sigma0', 'T_s', 'T_s_modis']
formula = model3.model.formula
print("---------------------------------------------------------------------")
print("Modelo planteado with backward selection and p-value: " + str(formula))
print("R^2 del modelo: " + str(model3.rsquared))
print("---------------------------------------------------------------------")



