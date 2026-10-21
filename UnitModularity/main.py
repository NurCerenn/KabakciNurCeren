# The work frame will look someting like follows: 
    #I will check the Temp data Pcpn data here first.
    #I will do clean-up to data if needed so that we can get better estimations
    #Then I will write scripts to analyse both datasets
        #climate variable vs year chart - slope
        #histogram of all the slopes
        #maps for each climate vaiable - colorful
#gather all results together and determine trends and see if the results concluding a climate change pattern

#MODULARITY EXAMPLE: Lesson 4, I decided on the interfaces at the begining before started any coding.
# I know it is not excatly precise with inputs and outputs, but I believe I'll get there as I have more practice.

#general pseudo code: (MODULARITY EXAMPLE: Lesson 6, write pseudocode)
#FOR each climate variable (temperature, precipitation):
#    raw      = load the file
#    clean    = remove unusable rows
#    slopes   = for each station with >= 40 measurements:
#                   regress value on year, keep the slope
#    draw histogram of slopes
#    draw map of slopes
# Interpret both results

#MODULARITY EXAMPLE: Lesson 5, it took me some time to even get the pseudo code shaped in my mind,
#but once I understood what I need to do, it was eaier to follow.

#Module 1: load datasets and take a look
# before we can take a look I need to learn how to work with rds files.


import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import metrics
from sklearn import tree
import pyreadr
import math
import scipy as sp

#MODULARITY EXAMPLE: Lesson 9 and 10, I wrote everything in the main.py first but then decided on splitting it into separate scripts.

from read_data import load_data, clean_data, check_data

#Percipitation data
read_p=load_data('UnitModularity/USAAnnualPcpn1950_2008.rds')
check_data(read_p, title='precipitation')

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

#looks like "data" has huge missing cells in Pcpn dataset, 
# I think best is to get rid of it as a whole and see the ways to manage the "lon" and "lat" columns.

#the goal is to determine wheather there's a climate change based on the tamp and precp data in USA


# Temperature data

read_t=load_data('UnitModularity/USAAnnualTemp1950_2008.rds')
check_data(read_t, title='temperature')

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

p_clean=clean_data(read_p)
check_data(p_clean, title= 'precipitation')

t_clean=clean_data(read_t)
check_data(t_clean, title='temperature')

# now we can move to next module since we cleaned up the datasets

# Module 3 - Slopes
# slopes   = for each station with >= 40 measurements:
#                   regress value on year, keep the slope
# identify stations

t_clean['name'].nunique()
t_clean[['state','name']].drop_duplicates().shape[0]
t_clean[['state','name','lon','lat']].drop_duplicates().shape[0]
t_stations=t_clean.groupby(['state','name']).size()
print((t_stations>=40).sum())

p_clean['name'].nunique()
p_clean[['state','name']].drop_duplicates().shape[0]
p_clean[['state','name','lon','lat']].drop_duplicates().shape[0]
p_stations=p_clean.groupby(['state','name']).size()
print((p_stations>=40).sum())


#MODULARITY EXAMPLE: Initially I had station identification, gathering slopes and computing all slopes
#in one module and then realized that they can be their own modules. It was Lesson 7 for me.
#
# Module 4 - gather slopes
# y = a + b * year, b is the slope = covariance(year, value) / varience(year)
from scipy.stats import linregress
from scipy import stats

from analysis import slope_each_station

# lets test it quickly for one slope
t_groups=t_clean.groupby(['state','name'])
test1=t_groups.get_group(t_stations.idxmax())
print(len(test1))
print(type(t_groups))
print(t_stations.head())
print(slope_each_station(test1))

# Module 5 - to compute all slopes

from analysis import compute_all_slopes

t_slopes=compute_all_slopes(t_clean)
p_slopes=compute_all_slopes(p_clean)

print(len(t_slopes), len(p_slopes))
print(t_slopes.head())
print(p_slopes.head())

#Module 6 - historams
# now I calculated and stored slopes for each station, let's see what they look like together 
# and what does the data stating about climate change?

#MODULARITY EXAMPLE: Lesson 8, I wrote notes for myself to remind what each chunks are for.

from analysis import plot_histogram
import matplotlib.pyplot as plt


plot_histogram(t_slopes,'temperature')
plot_histogram(p_slopes,'precipitation')

print(t_slopes['slope'].mean())
print(p_slopes['slope'].mean())

# Module 7 - mapping

from analysis import plot_map


plot_map(t_slopes,'temperature')
plot_map(p_slopes,'precipitation')

#MODULARITY EXAMPLE: Lesson 3, I did a lot of back and forth 
# in order to tune my script before I could get workable results
# 
# MODULARITY EXAMPLE: I tuned MY CODE FURTHER, wrote the functions into separate scriptscalled them in the main.py.
#  it s actually really fun to write code in this way, much easier.

# this is my attempt to write everythig into a single loop

import read_data
import analysis

for path, label in [("UnitModularity/USAAnnualTemp1950_2008.rds", "temperature"),
                    ("UnitModularity/USAAnnualPcpn1950_2008.rds", "precipitation")]:
    df = read_data.load_data(path)
    clean = read_data.clean_data(df)
    slopes = analysis.compute_all_slopes(clean)
    print(label, "stations:", len(slopes), "mean slope:", slopes["slope"].mean())
    analysis.plot_histogram(slopes, label)
    analysis.plot_map(slopes, label)

#oh wow, it worked! WOW THIS IS REALLY COOL! I spent a lot of time writing the code 
# and as I keep working on it my code got shorther and shorther. Wow. I'm glad that I enrolled in this class. Thank you.