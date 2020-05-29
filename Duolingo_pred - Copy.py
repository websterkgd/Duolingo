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
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#change directory to directory with data
os.chdir('D:\\a_Desktops_Git\\Current\\SpringBoard\\Capstone')

#import the data 
duod = pd.read_csv('settles.acl16.learning_traces.13m.csv') #takes 2 minutes

#d_li = pd.get_dummies(duod['lexeme_id'], sparse=True) #this works #too much memory
#d_ui = pd.get_dummies(duod['user_id'], sparse=True) #this works #too much memory

#may want to find faster ways of analyzing the data
# may want to try importing in chunks

#pull out how many users
lu = duod.user_id.values
lu = list(set(lu)) # runs quickly len 115,222

# create a list of dictionaries
# currently loading 45 users in first minute .. so will say 40 per minute
# need to reindex for future analytical steps
ldu_oh = {}
for i in list(range(0,14400)):
    ldu_oh[lu[i]] = duod[duod.user_id == lu[i]]
    ldu_oh[lu[i]].index = list(range(0,len(ldu_oh[lu[i]])))
    print(i)

#deleting entries in ldu_oh with less than one-hundred rows reduction by 2/3
ldu_ohr = {}    
for key in ldu_oh:
    if len(ldu_oh[key]) > 100:
        ldu_ohr[key] = ldu_oh[key]

print(len(ldu_ohr)) # 15 users with 100 entries or greater

#ldu_ohr['u:gaCJ']

#examining total_seen vs time to figure out how to create metric
#for key in ldu_ohr:
#    plt.scatter(ldu_ohr[key]['timestamp'],ldu_ohr[key]['history_seen'], 
#            alpha =0.3, c = 'g')
#    plt.show()

# not a simple relationship some users would work.. others are difficult

# this simple analysis has the complication of seeing differing lexemes
# I could count unique timestamps
    
# I like this idea and will try it
    

# this is working. Is there anything else I want to do before creating 
# a regression line?

# just trying to plot relationship for all users
#for key in ldu_ohr:
#    ut = list(set(ldu_ohr[key].timestamp))
#    ut.sort()
#    ti = list(range(0,len(ut)))
#    plt.scatter(ut,ti, alpha =0.8, c = 'g')
#    plt.show()

# now to run regressions and capture max ti, duration, r2, languages, slope

#create dataframe
dm = pd.DataFrame(np.zeros([len(ldu_ohr),5]))
dm.columns =['max_ti','dur','r2','slope','ltl']    
dm.index = ldu_ohr.keys()

from scipy import stats
for key in ldu_ohr:
    ut = list(set(ldu_ohr[key].timestamp))
    ut.sort()
    ti = list(range(0,len(ut)))
    dm.max_ti.loc[key] = max(ti)
    dm.dur.loc[key]  = max(ut)-min(ut)
    dm.slope.loc[key]  = stats.linregress(ut, ti)[0]
    dm.r2.loc[key]  = stats.linregress(ut, ti)[2]**2
    dm.ltl.loc[key] =  ldu_ohr[key].ui_language[0]+ldu_ohr[key].learning_language[0]

print(dm.head)


#Yay I have a dataframe
    
##write dm to a txt file
#dm.to_csv('duo_ui.csv') #the range on this file was from 0 to 14400



######### HERE I THINK 

# I don't know exactly where I am on this analysis. But I want to create an
# intensity of use vector to describe each user

# I think intensity can be described by 3 metrics - 
# 1 the slope of the use 
# 2 the total duration of the use
# 3 the r2 of the use

# I then want to predict intensity of use based on language learning pairs








#can I predict the probability that a user will get an answer right?

#Can I predict p? and to what degree 

# paper uses half life regression - new technique 
# how can I do? 

# first I just want to use some of the techniques I've already learned

# p correct for lexeme = f(user)#may exclude, f(time since practice)  
# f(string length) #f (times practiced) # i expect these terms to interact
# time since practice is likely not linear, but I will try a linear model






# 
# X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].valuesy = dataset['quality'].values
# 
# plt.figure(figsize=(15,10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['quality'])
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train)
# 
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
# coeff_df
# 
# y_pred = regressor.predict(X_test)
# 
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})df1 = df.head(25)
# 
# df1.plot(kind='bar',figsize=(10,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()
# 
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# =============================================================================


# previous code

## create column of string length for each user
#for key in ldu_ohr:
#    ldu_ohr[key]['sl'] = pd.Series(list(range(0,len(ldu_ohr[key]))))
#    for i in list(range(len(ldu_ohr[key]))):
#        ldu_ohr[key]['sl'][i] = len(ldu_ohr[key].lexeme_string[i].split("/", 1)[0])
#        
##the time vector is delta
#
##p_correct is the proportion of lexeme right in a session 
#
## p correct for lexeme = f(user)#may exclude, f(time since practice)  
## f(string length) #f (times practiced) # i expect these terms to interact
## time since practice is likely not linear, but I will try a linear model
#
## predictions of p using string length, times practiced, and time since 
## last practice per user
#
#po = [] #observed probability
#pp = [] #predicted p
#
#for key in ldu_ohr:
#    X = ldu_ohr[key][['delta','history_seen','sl','session_seen']]
#    y = ldu_ohr[key]['p_recall']
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
#                                                         random_state=42)
#    regressor = LinearRegression()  
#    regressor.fit(X_train, y_train)
#    y_pred = regressor.predict(X_test)
#    po = po + list(y_test)
#    pp = pp + list(y_pred)
#
#plt.scatter(po, pp, alpha =0.1, c = 'g')
#plt.show()
#
##this model doesn't work very well
##large scatter around observed p of 1, never predicts p of 0 
#
##adding session seen doesn't improve model much but now model predicts below 0.5
##still large scatter around 0.1
#
##simple regression model 
#
#X = pd.DataFrame([duod[['delta','history_seen','session_seen']],d_li,d_ui])
#y = duod['p_recall']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
#                                                         random_state=42)
#regressor = LinearRegression()  
#regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_test)
#df = pd.DataFrame({'Obs': y_test, 'Pred': y_pred})
#
#plt.scatter(df['Obs'],df['Pred'], alpha =0.005, c = 'g')
#plt.show()

