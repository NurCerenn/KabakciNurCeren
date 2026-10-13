#This script is to read and clean the data for the main.py 
# this way if there's any other data set we can add into our analysis we can just load it and use it in the main.py

#MODULARITY EXAMPLE: Lesson 9 and 10, I wrote everything in the main.py first but then decided on splitting it into separate scripts.

import pandas as pd
from pandas import DataFrame
import numpy as np
import pyreadr



def load_data(path):
    ''''Input: file path (text).
       Output: DataFrame read from the .rds file.'''
    return pyreadr.read_r(path)[None]

def clean_data(df):
    '''Input: dataframe
    output: dataframe without NaN in the data'''
    return df.dropna()

def check_data(df, title):
    print('===', title, '===')
    print('type:', type(df))
    print('rows, columns:', df.shape)
    print('column names:', df.columns.tolist())
    print('missing values per colunm:')
    print(df.isna().sum())
    print(df.head)
    print()