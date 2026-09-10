import numpy as np
import pandas as pd
import math
import scipy as sp
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt

#load the data
bc=pd.read_csv("BreastCancerData.csv")
print(bc.head)
print(type(bc))

bc=bc.iloc[:,1:12]
bc.columns=["diagnosis","radius","texture","perimeter","area","smoothness",
           "compactness","concavity","concave_points","symmetry",
           "fractal_dimension"]
print(bc.shape)
print(bc.columns)
print(type(bc))

#split test and validation sets
np.random.seed(101)
bc_per=bc.iloc[np.random.permutation(len(bc)),:]
print(bc_per.shape)
print(bc_per.columns)
bc_val=bc_per.iloc[0:(math.floor(.75*len(bc_per))),:]
bc_test=bc_per.iloc[(math.floor(.75*len(bc_per))):(len(bc_per)+1),:]

#consistancy checks
print(len(bc_val)+len(bc_test))
print(len(bc_per))  #check if their lenght are consistent with each other

Xval=bc_val.iloc[:,1:11] #this .iloc lets you to select rows or columns
yval=bc_val.diagnosis 

Xtest=bc_test.iloc[:,1:11] 
ytest=bc_test.diagnosis


#fit a CART
m_d=tree.DecisionTreeClassifier(max_depth=30,min_samples_split=20,
                                min_samples_leaf=7,
                                min_impurity_decrease=0.01,
                                max_features=None,
                                random_state=101)
print(type(m_d))
m_d.fit(Xval,yval)

#plot a picture of it
feature_names=bc_val.columns[1:12].to_list()
plt.figure()
tree.plot_tree(m_d,filled=True,feature_names=feature_names)
plt.show()