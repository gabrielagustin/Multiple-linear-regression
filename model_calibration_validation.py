# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018
@author: gag 
"""

import pandas as pd
import statsmodels.formula.api as smf
import selection
import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura
import copy
import seaborn as sns




sns.set_style("whitegrid")

file = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/mediciones_sensores_CONAE_MonteBuey_SMAP/SM_CONAE_Prom/extract_table_2.csv"

data = lectura.lecturaCompleta_etapa1_SAR_SMAP(file)


## se mezclan las observaciones de las tablas
## semilla para mezclar los datos en forma aleatoria

rand = 0
np.random.seed(rand)
dataNew = selection.shuffle(data)
dataNew = dataNew.reset_index(drop=True)

## formula

formula = "SM_CONAE ~ 1+Ts_SMAP+GPM+sigma0_vv_"

print("Modelo planteado:" + str(formula))
model2 = smf.ols(formula, dataNew).fit()
print("R^2 del modelo: " + str(model2.rsquared))


####### obtencion automatica de los porcentajes de datos
###### para entrenar y probar

print("Obtencion automatica de los porcentajes de datos")
type = "RMSE"
#type = "R^2"
var = "RSOILMOIST"
porc = 0.75
print("Porcentaje de datos de calculo: " + str(porc))


#### division de los datos para entrenamiento y prueba
dataTraining, dataTest = train_test_split(data, test_size=0.25)

print("-------------------------------------------------------------------")
print(dataTraining)
print(dataTraining.describe())
print("-------------------------------------------------------------------")
print(dataTest)
print(dataTest.describe())
print("-------------------------------------------------------------------")


#### Entrenamiento
print("Entrenamoiento: ")
MLRmodel = smf.ols(formula, dataTraining).fit()
print(MLRmodel.summary())
print("R^2 del modelo: " + str(MLRmodel.rsquared))


#### error de calibracion
xxx = copy.copy(dataTraining)
del xxx['SM_CONAE']
yTraining = dataTraining['SM_CONAE']
yCal = MLRmodel.predict(xxx)
yTraining = 10**(yTraining)
yCal = 10**(yCal)

rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
print("RMSE:" + str(rmse))
bias = statistics.bias(yTraining,yCal)
print("Bias:" + str(bias))


#print "RMSE del modelo: " + str(np.sqrt(model.mse_resid))

#### se guardan los coeficientes del modelo entrenado
print("Los coeficientes del modelo son: ")
coeff =  MLRmodel.params
#print coeff[1]

print("Calculo de los VIF: ")
print("Orden de las variables")
print(list(dataNew))
matrix = np.array(dataTraining)
vifs = statistics.calc_vif(matrix)
print(vifs)
#vifs = aca.variance_inflation_factor(matrix,2)
#print vifs
#### prueba del modelo

#### Prueba
print("Prueba: ")

y = np.array(dataTest["SM_CONAE"])

pred = MLRmodel.predict(dataTest)
yAprox = np.array(pred)
#### aca!!!
#y = np.exp(y)
#yAprox = np.exp(yAprox)

y = 10**(y)
yAprox = 10**(yAprox)

bias = statistics.bias(yAprox,y)
print("Bias Validacion:" + str(bias))

print("Rango real: "+ str(np.max(y))+"..." + str(np.min(y)))
print("Rango aproximado: "+ str(np.max(yAprox))+"..." + str(np.min(yAprox)))


## se obtiene el error
rmse = 0
rmse = statistics.RMSE(y,yAprox)
print("RMSE:" + str(rmse))
RR = sklearn.metrics.r2_score(y, yAprox)
print("R2:" + str(RR))
error = np.zeros((len(y),1))
#for i in range(0,len(error)):
error = np.abs(y-yAprox)


d = {'y': y, 'yAprox': yAprox}
df = pd.DataFrame(data=d)
RR = smf.ols('y ~ 1+ yAprox', df).fit().rsquared
print("R^2 222: "+str(RR))


df = pd.DataFrame({'yTraining':yTraining,
                   'yCal':yCal,
                   })

#df.loc[df.yCalMLP==1, 'yCal'] *= 2

#sns.pairplot(data=df,
            #x_vars=['yTraining'],
            #y_vars=['yCal', 'yCalMLP'])

fig = plt.figure(1,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="yTraining", y="yCal", marker="+", fit_reg=True, data=df, scatter_kws={'s':50}, label='MLR', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Scatterplot for the Association between Breast Cancer and Female Employment');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')




df = pd.DataFrame({'y':y,
                   'yAprox':yAprox,
                   })



#dataNew = df[(df.y < 44)]
#df = dataNew


fig = plt.figure(2,facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(5,50)
ax.set_ylim(5,50)
sns.regplot(x="y", y="yAprox", marker="+", fit_reg=True, data=df, scatter_kws={'s':50}, label='MLR', ax=ax)
plt.xlabel('Observed value [% Vol.]', fontsize=12);
plt.ylabel('Estimated value [% Vol.]', fontsize=12);
#plt.title('Scatterplot for the Association between Breast Cancer and Female Employment');
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')
plt.grid(True)






plt.show()


