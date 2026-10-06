#I will check the data Pcpn data here first.
#load it and take a look

#before we can take a look I need to now how to work with rds files.

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import metrics
from sklearn import tree
import pyreadr
import math
import scipy as sp


read=pyreadr.read_r("UnitModularity/USAAnnualPcpn1950_2008.rds")
print(read.keys())
print(type(read))

#I need to convert rds to pandas dataframe in order to work in py environment

df=read[None]
print(type(df))

#now lets explore out data
# What type of object is this? -> converted to df
# How big is it? -> (152869, 6)
# What are the columns and their types? 
# --Index(['state', 'name', 'lon', 'lat', 'data', 'year'], dtype='str')
# --state    category
# --name     category
# --lon       float64
# --lat       float64
# --data      float64
# --year        int32
# What's missing? 
# -- [152869 rows x 6 columns]
#    state         0
#    name          0
#    lon        1239
#    lat        1239
#    data     126630
#    year          0
#    dtype: int64

print(df.shape)
print(df.head())
print(df.tail())
print(df.columns)
print(df.dtypes)

print(df.isna())
print(df.isna().sum())

#looks like "data" has huge missing cells, I think best is to get rid of it as a whole and see the ways to manage the "lon" and "lat" columns.