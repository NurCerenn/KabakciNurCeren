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


# data load and screening

columns = [
    "sex", "length", "diameter", "height",
    "whole_weight", "shucked_weight", "viscera_weight",
    "shell_weight", "rings",
]

data=pd.read_csv('Unit05_RegressionEtc/AbaloneData.csv', header=0, names=columns)
print(data['sex'].unique())
print("dtypes:", data.dtypes)

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


cols_info=[data.count]
plt.hist(cols_info)

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
hist_each_column(data, columns)


corr_matrix=data.corr(numeric_only=True)
print("correlation:", corr_matrix)
print("Avarage correlation of each column with others:")
print(corr_matrix.abs().mean().sort_values())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()



# Model Building

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

print("my_tree CV rmse:", cv_rmse(mytree))
print("full_tree CV rmse:", cv_rmse(tree_full))

print("pruned tree rmse in-sample-error:", rmse1)
print("full tree rmse in-sample-error:", rmse_full)
#my tree CV rmse: 2.587337228445744
#my tree rmse in-sample-error: 2.5573856798270596

#full tree CV rmse: 3.058829379409535
#full tree rmse in-sample-error: 0.0

# is the out-of-sample rmse higher or lower than the within-sample rmse for each model? 
#       For pruned tree(my_tree), the in-sample and the out-of-sample errors are pretty close to each other,
#      where for the full tree in sample error is 0.0 and the out of sample error is higher than the pruned tree.

# Which model is better within-sample?
#       The full tree is better within-sample (0.0 vs 2.6)

# which is better out-of-sample? 
#       The pruned tree is better out of sample (2.6 vs 3.1)

# What does this tell you about overfitting?
#       The full tree performs perfectly within-sample while performing poorly out-of-sample. It is overfitting.


#BEFORE PICKING A MODEL, lets do a bagging and RF, and see how they perform


# Bagging
from sklearn.ensemble import BaggingRegressor
m_bag= BaggingRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=500,
                            oob_score=True)

m_bag.fit(X_val,y_val)


#RF
from sklearn.ensemble import RandomForestRegressor
m_rf=RandomForestRegressor(n_estimators=1000, max_features=1/3)
m_rf.fit(X_val,y_val)

print('RF values:')
print(m_rf.get_params()['max_features'])
print(m_rf.n_features_in_) #How many predictor columns did you receive when I fitted you? -> 10
print(m_rf.estimators_[0].max_features_)



#scores:
#   out-of-sample
print("m_bag out-of-sample:", cv_rmse(m_bag))
print("RF m_bag out-of-sample:", cv_rmse(m_rf))

#within-sample error
rmse_bag = np.sqrt(mean_squared_error(y_val, m_bag.predict(X_val)))
rmse_rf = np.sqrt(mean_squared_error(y_val, m_rf.predict(X_val)))

print("m_bag within-sample:", rmse_bag)
print("RF m_bag within-sample:", rmse_rf)

# m_bag out-of-sample: 2.1828300247313566
# RF m_bag out-of-sample: 2.1499834336677233
# m_bag within-sample: 0.7982275464029829
# RF m_bag within-sample: 0.7881303585100047

# bagging anf RF performed almost the same


# lets fit a linear regression model

from sklearn.linear_model import LinearRegression

m_lin=LinearRegression()
m_lin.fit(X_val,y_val)

print('m_lin cross-validation:', cv_rmse(m_lin))
# m_lin cross-validation: 2.2550265122368334

# #my tree CV rmse: 2.587337228445744


# compared to bagging and RF, linear regression performed a little worse but it also performed better than pruned tree.
# It does not "win" but doesnt lose drasticly either. I guess it suggest that there is a linear relationship in this data and predictors.
# but stating it like this also doesn't feels right to me either, since I know the abolone data. 
# OR maybe it means that this data is NOT suitable for the linear models and we should evaluate other models.


# I choose RF model to move forward and test with the vault data, which performed the best for me

rf_pred=m_rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("RF test rmes:", rf_rmse)
print("RF test r2", rf_r2)



# predicted vs observed

plt.scatter(y_test, rf_pred, alpha=0.3)
lims=[y_test.min(), y_test.max()]
plt.plot(lims, lims, color="red")
plt.xlabel('Observed Rings (test data)')
plt.ylabel('Predicted Rings (test data)')
plt.title('RF predicted vs. observed, test set')
plt.show()


