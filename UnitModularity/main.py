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
    return df.dropna()

t_clean=clean_data(df_t)
p_clean=clean_data(df_p)


print(type(t_clean))
print(type(p_clean))
print(t_clean.columns)
print(t_clean.columns.tolist())
print(t_clean.index.names)
print(p_clean.columns)
print(p_clean.columns.tolist())
print(p_clean.index.names)

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

# gather slopes
# y = a + b * year, b is the slope = covariance(year, value) / varience(year)
from scipy.stats import linregress
from scipy import stats

def slope_each_station(station_df):  #this formula is to calculate a single slope for just one
    """Input: DataFrame with 'year' and 'data' for ONE station.
       Output: float, fitted slope (data units per year)."""
    result=linregress(station_df['year'],station_df['data'])
    return result.slope

# lets test it
t_groups=t_clean.groupby(['state','name'])
test1=t_groups.get_group(t_stations.idxmax())
print(len(test1))
print(type(t_groups))
print(t_stations.head())
print(slope_each_station(test1))

# to compute all slopes
def compute_all_slopes(df, min_n=40):
    """Input: cleaned table with many stations.
       Output: table with one row per station (state, name, lon, lat, slope, n)."""
    records=[] #to keep results

    for (state,name), station_df in df.groupby(['state','name']):
        n=len(station_df)
        if n<min_n:
            continue
        slope=slope_each_station(station_df)

        records.append({
            'state': state,
            'name': name,
            'lon': station_df['lon'].iloc[0],
            'lat': station_df['lat'].iloc[0],
            'slope': slope,
            'n': n,
        })

    return pd.DataFrame(records)

t_slopes=compute_all_slopes(t_clean)
p_slopes=compute_all_slopes(p_clean)

print(len(t_slopes), len(p_slopes))
print(t_slopes.head())
print(p_slopes.head())

#Module 4 - historams
# now I calculated and stored slopes for each station, let's see what they look like together 
# and what does the data stating about climate change?

import matplotlib.pyplot as plt

def plot_histogram(slopes_df, label):
    """Input: table with a 'slope' column, and a label (text) for the title.
       Output: draws a histogram."""
    plt.hist(slopes_df['slope'], bins=30)
    plt.axvline(0, color='red')
    plt.title("Slopes of "+label+'vs year')
    plt.xlabel('Slope (change per year)')
    plt.ylabel("# of stations")
    plt.show()
    

plot_histogram(t_slopes,'temperature')
plot_histogram(p_slopes,'precipitation')

print(t_slopes['slope'].mean())
print(p_slopes['slope'].mean())

# Module 5 - mapping

def plot_map(slopes_df, label):
    """Input: table with 'lon', 'lat', 'slope' columns, and a label (text).
       Output: saves a map as a .png file."""
    limit=slopes_df['slope'].abs().max()
    plt.scatter(slopes_df['lon'], slopes_df['lat'],
                c=slopes_df['slope'], cmap='coolwarm',
                vmin=-limit,vmax=limit, s=10)
    plt.colorbar(label='Slope (change per year)')
    plt.title('Slope of '+ label + ' vs year')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

plot_map(t_slopes,'temperature')
plot_map(p_slopes,'precipitation')