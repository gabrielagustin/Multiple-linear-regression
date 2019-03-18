# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018
@author: gag 

En este script se presentan diferentes métodos de selección de variables
para el método de regresión lineal múltiple. 

"""

import numpy as np
import statistics
import sklearn
import statsmodels.formula.api as smf

#import matplotlib.pyplot as plt
#import selection



def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    #print remaining
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            #print formula
            #score = smf.ols(formula, data).fit().rsquared_adj
            #print score
            score = smf.ols(formula, data).fit().rsquared
            #print score
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    print (model.summary())
    return model

def vif_forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    #print remaining
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            #print formula
            R2 =  smf.ols(formula, data).fit().rsquared
            score = 1/(1-R2)
            #score = smf.ols(formula, data).fit().rsquared
            #print score
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    return formula


def shuffle(df):
    index = list(df.index)
    np.random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df

#def calc_best_porcentaje(dataNew, formula, type, var):
    ##np.random.seed(0)
    #nRow = len(dataNew.index)
    #vPorc = np.arange(0.1, 0.95, 0.005)
    ##vPorc = np.arange(0.1, 0.9, 1/float(nRow))
    ##print vPorc
    #vError = np.zeros((len(vPorc), 1))
    #vErrorModel = np.zeros((len(vPorc), 1))

    #for i in range (0, len(vPorc)):
        #porc = vPorc[i]
        ##print porc
        #numTraining = int(np.round(nRow*porc,0))
        ##print "numero calibration: "+ str(numTraining)
        #numTest = nRow -numTraining
        ##print "numero validation: "+ str(numTest)
        #dataTraining =  dataNew.ix[:numTraining, :]
        ##print len(dataTraining.index)
        #dataTest = dataNew.ix[numTraining:, :]
        ##print len(dataTest.index)
        ##if ((numTraining > 20) and (numTest > 20)):
        #model = smf.ols(formula, dataTraining).fit();
        #y = np.array(dataTest[var])
        #y = 10**(y)
        #pred = model.predict(dataTest);
        #yAprox = np.array(pred)
        #yAprox = 10**(yAprox)
        ##yAprox = np.exp(yAprox)
        #if (type == "RMSE"):
            #pred = model.predict(dataTraining)
            #yCal = np.array(pred)
            #yCal = 10**(yCal)
            #y0 = np.array(dataTraining[var])
            #y0 = 10**(y0)
            #vErrorModel[i] = statistics.RMSE(y0, yCal)
            #vError[i] = statistics.RMSE(y, yAprox)
        #if (type == "R^2"):
            #vErrorModel[i] = model.rsquared
            #vError[i] = sklearn.metrics.r2_score(y, yAprox)

    #print vErrorModel
    #fig = plt.figure(1,facecolor="white")
    #fig0 = fig.add_subplot(111)
    #fig0.scatter(vPorc*100, vErrorModel,color = 'black', marker = "o", label='Calibration error')
    #fig0.scatter(vPorc*100, vError, color = 'blue', facecolors='none', marker = "o", label='Validation error')
    ##fig0.set_title(str(type))
    #fig0.set_xlabel("Set calibration size[%]",fontsize=12)
    #fig0.set_ylabel(str(type) +" "+"[% GSM]",fontsize=12)
    ##fig0.axis([9, 91, 4,7])
    #plt.xticks(np.linspace(10, 90, 15, endpoint=True))
    #plt.legend(loc=1, fontsize = 'medium')

    #minimo = 999
    #for i in range(0,len(vError)):
        #if (vError[i]!= 0):
            #if (np.abs(vError[i]-vErrorModel[i]) < minimo):
                #minimo = np.abs(vError[i] - vErrorModel[i])
                #indexMin = i
            ##fig0.text(10, 7.5, 'Perc=%5.3f' % (vPorc[indexMin]*100), fontsize=12)
    #plt.show()
    #return vError[indexMin], float(vPorc[indexMin])
#
