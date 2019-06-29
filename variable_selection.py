# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018
@author: gag 
"""


import selection
import numpy as np
import statistics
import lectura
import pandas as pd


#### Etapa 3

file = "DataSet_1km.csv"
#file = "DataSet_9km.csv"
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
dataNew = selection.shuffle(data)
dataNew = data.reset_index(drop=True)
dataNew = data


model1 = selection.forward_selected(data, 'RSOILMOIST')
formula = model1.model.formula
print "Modelo planteado with forward selection and Rsquare: " + str(formula)
print "R^2 del modelo: " + str(model.rsquared)



model2 = selection.forward_selected.vif(data, 'RSOILMOIST')
formula = model2.model.formula
print "Modelo planteado with forward selection and VIF: " + str(formula)
print "R^2 del modelo: " + str(model.rsquared)



statistics.stats(dataNew,'SM_SMAP')

matrix = np.array(dataNew)
print("Orden de las variables")
print(list(dataNew))
print(statistics.calc_vif(matrix))


