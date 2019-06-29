# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Oct 16 09:09:04 2018

@author: gag 

In this file, different methods of variable selection for the multiple linear regression method are presented.

"""

import numpy as np
import statistics
import sklearn
import statsmodels.formula.api as smf



def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept selected by forward selection
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
    # print (model.summary())
    return model


def forward_selected_vif(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept selected by forward selection
           evaluated by adjusted Variance Inflation Factor (VIF)
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
            #calculo VIF
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
    model = smf.ols(formula, data).fit()
    # print (model.summary())
    return model


def shuffle(df):
    index = list(df.index)
    np.random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df


