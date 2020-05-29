# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:48:36 2020

@author: websterkgd
"""

#clear environment
from IPython import get_ipython;   
get_ipython().magic('reset -sf')
from IPython import get_ipython; 

#import packages for data analysis 
import pandas as pd
import os 
import numpy as np  
import matplotlib.pyplot as plt  

#change directory to directory with data
os.chdir('D:\\a_Desktops_Git\\Current\\SpringBoard\\Capstone')

#import the data 
duod = pd.read_csv('settles.acl16.learning_traces.13m.csv') #takes 2 minutes

#pull out how many users
lu = duod.user_id.values
lu = list(set(lu)) # runs quickly len 115,222

#pulling about 60 user ~ 1.5 min (fine for prelim analysis) 
# create a list of dictionaries 
ldu_oh = {}
for i in list(range(0,60)):
    ldu_oh[lu[i]] = duod[duod.user_id == lu[i]]
    ldu_oh[lu[i]].index = list(range(0,len(ldu_oh[lu[i]])))
    print(i)

#deleting entries in ldu_oh with less than one-hundred rows reduction by 2/3
ldu_ohr = {}    
for key in ldu_oh:
    if len(ldu_oh[key]) > 100:
        ldu_ohr[key] = ldu_oh[key]

print(len(ldu_ohr)) # ~ 20 users

#i want to predict p_recall - defined as session_correct/session_seen
#quick plotting to verify
#for key in ldu_ohr:
 #   plt.scatter(ldu_ohr[key].session_correct/ldu_ohr[key].session_seen, 
  #              ldu_ohr[key].p_recall, alpha =0.7)  
   # plt.show()

#import models for linear reg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols

#hpp = ols('PRICE ~ PTRATIO',bos).fit()
#print(hpp.summary())

for key in ldu_ohr:
    lb_make = LabelEncoder()
    ldu_ohr[key]['cgy'] = lb_make.fit_transform(ldu_ohr[key].lexeme_id)
    m = ols('p_recall ~ delta +history_seen + cgy', ldu_ohr[key]).fit()
    print(m.summary())

#not a great model

#how do I implement half-life regression? 

#what else could I ask in the mean time?
    


###trying to predict p_recall for each lexeme 'delta' , 'lexeme_id',  
#for key in ldu_ohr:
#    lb_make = LabelEncoder()
#    ldu_ohr[key]['cgy'] = lb_make.fit_transform(ldu_ohr[key].lexeme_id)
#    X = ldu_ohr[key][['delta','history_seen','cgy']]
#    y = ldu_ohr[key]['p_recall']
#    regressor = LinearRegression()  
#    regressor.fit(X, y)
#    ldu_ohr[key]['ypr'] = regressor.predict(X)
#    plt.scatter(ldu_ohr[key].ypr, ldu_ohr[key].p_recall, alpha =0.7)  
#    plt.show()


