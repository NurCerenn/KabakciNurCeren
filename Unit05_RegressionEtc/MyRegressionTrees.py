# For this assignment, I choose to work with thw Abalone data. 
# The general plan is to first study and understand the data I'm working with, 
# do neccessary cleaning and modifications. 
# Then work on modelling: Data split, regression trees both defoult and a full tree. 
#       - Make a plot of predicted against observed values 
#       - Compute the RMSE and R^2 of the predictors
#       - Perform k-fold cross validation, using RMSE as the measure of out-of-sample error
#       - Improve on the single trees (pick one method)
#       - Fit an ordinary multiple linear regression to the data, do k-fol cross validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Module 1: data load and screening

columns = [
    "sex", "length", "diameter", "height",
    "whole_weight", "shucked_weight", "viscera_weight",
    "shell_weight", "rings",
]

data=pd.read_csv('Unit05_RegressionEtc/AbaloneData.csv', header=0, names=columns)
print(data['sex'].unique())
print("dtypes:", data.dtypes)
#convert_dtypes={"length":float, "diameter":float, "height":float,
    #"whole_weight":float, "shucked_weight":float, "viscera_weight":float,
    #"shell_weight":float, "rings":float}
#data=data.astype(convert_dtypes)
data['sex']=data['sex'].astype('category')


print("dtypes:", data.dtypes)

# fist, let's get a good undertanding what this data looks like
print('type:', type(data))
print('rows, columns:', data.shape)
print('info:', data.info)
print('column names:', data.columns.tolist())
print('missing values per colunm:')
print(data.isna().sum())
print(data.head)

print(data.describe())
print(data['sex'].value_counts())
print("number columns:", (data.select_dtypes('number')==0).sum())
print("dtypes:", data.dtypes)

# let's plot  histograms


#cols_info=[data.count]
#plt.hist(cols_info)

def hist_of_columns(data, bins=30):
    columns=data.select_dtypes('number').columns
    for name in columns:
        plt.hist(data, bins=bins)
        plt.title(name)
        plt.show()



def hist_each_column(data, column, bins=30):
    '''for each colunm, make a histogram of that colunm and show it'''
    
    for name in column:
        plt.hist(data[name], bins=bins)
        plt.title(name)
        plt.xlabel(name)
        plt.ylabel('counts')
        plt.show()

column=data.select_dtypes('number').columns
#hist_each_column(data, columns)

corr_matrix=data.corr(numeric_only=True)
print("correlation:", corr_matrix)
print("Avarage correlation of each column with others:")
print(corr_matrix.abs().mean().sort_values())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()



# Module2: Model Building

from sklearn import tree
X=pd.get_dummies(data.drop(columns='rings'), columns=['sex'])
y=data['rings']

from sklearn.model_selection import train_test_split, cross_val_score
X_val, X_test, y_val, y_test=train_test_split(X,y, test_size=0.25, random_state=42)

print(X_val.shape)
print(y_val.shape)
print(X_val.head())
print(y_val.head())
print(X_val['shell_weight'].shape)

mytree=tree.DecisionTreeRegressor(max_depth=2,min_samples_split=20,
                                  min_samples_leaf=7,ccp_alpha=0.01,random_state=101)

mytree.fit(X_val,y_val)
print(mytree.get_depth(),mytree.get_n_leaves())
print(tree.export_text(mytree, feature_names=list(X_val.columns)))

# lets print our tree
from sklearn.tree import plot_tree, export_text

plt.figure(figsize=(16,8))
plot_tree(mytree, feature_names=X.columns, filled=True)
plt.show()

pred1=mytree.predict(X_val)

plt.scatter(y_val, pred1, alpha=0.3)
lims=[y_val.min(), y_val.max()]
plt.plot(lims, lims, color='red')
plt.xlabel('observed rings')
plt.ylabel('predicted rings')
plt.show()

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

rmse1 = np.sqrt(mean_squared_error(y_val, pred1))
r2_1 = r2_score(y_val, pred1)

tree_full = tree.DecisionTreeRegressor(
    max_depth=None, min_samples_leaf=1, ccp_alpha=0.0, random_state=42
)
tree_full.fit(X_val, y_val)
rmse_full = np.sqrt(mean_squared_error(y_val, tree_full.predict(X_val)))

from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model):
    scores = cross_val_score(model, X_val, y_val, cv=kf,
                             scoring="neg_root_mean_squared_error")
    return -scores.mean()

print("my tree CV rmse:", cv_rmse(mytree))
print("full tree CV rmse:", cv_rmse(tree_full))