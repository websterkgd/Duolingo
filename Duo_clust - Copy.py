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
dm = pd.read_csv('duo_ui.csv') 

#pulling out intensity metrics
im = dm[['max_ti','dur','r2','slope']]

#pulling out possible clustering variable
c = dm['ltl']

#clustering
#import pakages for PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

# normalize data
nim = pd.DataFrame(preprocessing.scale(im),columns = im.columns) 

#transpose data
tnim = pd.DataFrame.transpose(nim) 

#fit the data
pca = PCA()

pca.fit_transform(tnim)
#pcad = pd.DataFrame(pca.components_,columns=nim.columns,index = ['PC-1','PC-2'])

#plot prelabeling and after labeling
plt.scatter(pca.components_[0,:],pca.components_[1,:],alpha = 0.1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of language learning intensity')
plt.show()

#not a whole lot of clustering

#create numeric coding for color on plot
c_c = c.astype('category').cat.codes

#color using c
plt.scatter(pca.components_[0,:],pca.components_[1,:],
            alpha = 0.5, c = c_c)
plt.title('PCA of language learning intensity')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

#not a lot of seperation by language learning pairs

#plot variance explained diagram
#plot the diagram
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.title('Variance explained per PCA axis')
plt.xlabel('PCA axis')
plt.ylabel('variance')
plt.show()

#try 3d plotting
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca.components_[0,:],pca.components_[1,:],pca.components_[2,:],
            alpha = 0.5, c = c_c)
plt.show() #just a giant cluster

#affinity propagation may be the best method to use
#clustering via affinity propagation
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(nim) #started at 1:30 pm 1 min run time

#predicts 97 clusters when there should be 8 - 

#plot clusters on top of PCA
plt.scatter(pca.components_[0,:],pca.components_[1,:],
            c=clustering.labels_, alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('User intensity PCA, Affin. Prop. Clusters')
plt.show()

#trying k means for fun
from sklearn.cluster import KMeans
k7 = KMeans(n_clusters=7,random_state=42*42)

# Fit the kmeans to the samples
k7.fit(nim)
plt.scatter(pca.components_[0,:],pca.components_[1,:],
            c=k7.labels_, alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('User intensity PCA, K means. Clusters')
plt.show()

# k means performs poorly as expected ...

#need to try naive bayes and svm

#as well as RF and GBT

# Import necessary modules for svm
from sklearn import svm
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = nim
y = c_c

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=42, 
                                                    stratify= y)


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#check accuracy 
print(clf.score(X_test, y_test)) # 29.8 % accurate

#trying RBF kernal
#Create a svm Classifier
clf2 = svm.SVC(kernel='rbf') # rbf Kernel

#Train the model using the training sets
clf2.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf2.predict(X_test)

#check accuracy 
print(clf2.score(X_test, y_test)) # 30.2 % accurate

#trying poly kernal
#Create a svm Classifier
clf3 = svm.SVC(kernel='poly') # rbf Kernel

#Train the model using the training sets
clf3.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf3.predict(X_test)

#check accuracy 
print(clf3.score(X_test, y_test)) # 31.3 % accurate 
#model is 2.5 times more accurate than guessing 
#this could be due to an uneven distribution in language learning

#is this due to differences in users?
len(c_c[c_c==1])/len(c_c) # 0.27
len(c_c[c_c==2])/len(c_c) # 0.14
len(c_c[c_c==0])/len(c_c) # 0.10
len(c_c[c_c==3])/len(c_c) # 0.06
len(c_c[c_c==4])/len(c_c) # 0.02
len(c_c[c_c==5])/len(c_c) # 0.27
len(c_c[c_c==6])/len(c_c) # 0.03
len(c_c[c_c==7])/len(c_c) # 0.07

#use a confusion matrix
from sklearn.metrics import confusion_matrix
labels = list(set(c_c))
cm = confusion_matrix(y_test, y_pred, labels)
print(cm) #model is essentially just guessing everthing is a 1 or 5 