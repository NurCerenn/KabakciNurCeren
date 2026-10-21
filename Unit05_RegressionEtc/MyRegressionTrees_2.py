import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = [
    "sex", "length", "diameter", "height",
    "whole_weight", "shucked_weight", "viscera_weight",
    "shell_weight", "rings",
]
data=pd.read_csv('Unit05_RegressionEtc/AbaloneData.csv', header=None, names=columns)

data['sex'].astype('category')
print(data.dtypes)
print(data.describe())
print(data.columns)
#print(data.select_dtypes(include='object').columns)

from sklearn.preprocessing import OneHotEncoder
#encoded=OneHotEncoder(sparse_output=False).fit_transform(data['sex'])
#print(encoded)