# The work frame will look someting like follows: 
    #I will check the Temp data Pcpn data here first.
    #I will do clean-up to data if needed so that we can get better estimations
    #Then I will write scripts to analyse both datasets
        #climate variable vs year chart - slope
        #histogram of all the slopes
        #maps for each climate vaiable - colorful
#gather all results together and determine trends and see if the results concluding a climate change pattern

#general pseudo code:
#FOR each climate variable (temperature, precipitation):
#    raw      = load the file
#    clean    = remove unusable rows
#    slopes   = for each station with >= 40 measurements:
#                   regress value on year, keep the slope
#    draw histogram of slopes
#    draw map of slopes
# Interpret both results

#Module 1: load datasets and take a look
# before we can take a look I need to learn how to work with rds files.

#Percipitation data

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import metrics
from sklearn import tree
import pyreadr
import math
import scipy as sp


read_p=pyreadr.read_r("UnitModularity/USAAnnualPcpn1950_2008.rds")
print(read_p.keys())
print(type(read_p))

#I need to convert rds to pandas dataframe in order to work in py environment

df_p=read_p[None]
print(type(df_p))

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

print(df_p.shape)
print(df_p.head())
print(df_p.tail())
print(df_p.columns)
print(df_p.dtypes)

print(df_p.isna())
print(df_p.isna().sum())
#print(df_p.info())

#looks like "data" has huge missing cells in Pcpn dataset, 
# I think best is to get rid of it as a whole and see the ways to manage the "lon" and "lat" columns.

#the goal is to determine wheather there's a climate change based on the tamp and precp data in USA


# Temperature data

read_t=pyreadr.read_r("UnitModularity/USAAnnualTemp1950_2008.rds")
print(read_t.keys())
print(type(read_t))

#I need to convert rds to pandas dataframe in order to work in py environment

df_t=read_t[None]
print(type(df_t))


print(df_t.shape)
print(df_t.head())
print(df_t.tail())
print(df_t.columns)
print(df_t.dtypes)

print(df_t.isna())  # indicating missing values
print(df_t.isna().sum())
#print(df_t.info())

# --Index(['state', 'name', 'lon', 'lat', 'data', 'year'], dtype='str')
# --state    category
# --name     category
# --lon       float64
# --lat       float64
# --data      float64
# --year        int32

# [957865 rows x 6 columns]
# state         0
# name          0
# lon       18290
# lat       18290
# data     640335
# year          0
# dtype: int64

# Module 2 - data clean-up 
    # I will use .dropna for the data clean-up and remove the rows with missing datapoints for each dataset.
    # this will help me to observe changes meaningfully

def clean_data(df):
    return df.dropna(subset=['data','lon','lat'])

t_clean=clean_data(df_t)
p_clean=clean_data(df_p)

print(len(t_clean),len(p_clean))
print(t_clean.isna().sum())
print(p_clean.isna().sum())

print(t_clean.notna().sum())
print(p_clean.notna().sum())

# now we can move to next module since we cleaned up the datasets

# Module 3 - Slopes
# slopes   = for each station with >= 40 measurements:
#                   regress value on year, keep the slope

