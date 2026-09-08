#I'm exploring anticancer peptides dataset in thi file
#below I copied the packages import lines from ReumanCode_CART
#I'm already familiar with packages install and import steps and used them a lot previously with Jupyter-lab



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

data=pd.read_csv("ACPs_Breast_cancer.csv")
print(type(data))
print(data.head)
print(data.columns)
print(data.describe)
print(data.shape)
print(data.attrs)
print(data.tail)