# in this code I will try to classify antimicrobial peptides
# my data set is S.mutans_AMP_dataset_v2_equal_1-0_(all_phschem
# _props).csv
# I generated this data set by using DBAASP database
# this dataset includes AMPs and non-AMPs sequences and 
# their physiochemical properties.

# setting initial libraries I will use

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import math 

# read the csv file into a dataframe
amp_pred= pd.read_csv("S.mutans_AMP_dataset_v2_equal_1-0_(all_phschem_props).csv")
print(amp_pred)
print(amp_pred.columns)
print(amp_pred.head())
print(amp_pred.shape)

# dataset clearence: removing columns we won't use for classification, which have 
# null celss

amp_pred.pop('C TERMINUS')
amp_pred.pop('SYNTHESIS TYPE')
amp_pred.pop('MIC')
amp_pred.describe



# data split into val and test 

# notes: I'm used to using sklearn's train_test_split. But I also went over 
# 'ReumanCode_CART.py' file and spent my time understanding manual splitting 

from sklearn.model_selection import train_test_split

X=amp_pred.drop(columns=['SEQUENCE', 'AMP Class'])
y=amp_pred['AMP Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101, stratify=y
)

# sanity check, note: value_counts(normalize=True) shows the proportion of each class
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


from sklearn  import tree
from sklearn.ensemble import RandomForestClassifier

AMP_tree=tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,
                                     min_samples_split=20, random_state=101,
                                     min_impurity_decrease=0.01)

print(type(AMP_tree))
AMP_tree.fit(X_train,y_train)



# visualize
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize = (25, 15), dpi = 80)
plot_tree(AMP_tree, feature_names = X.columns, class_names = y.unique().astype(str), filled=True, rounded=True)
plt.title('Decision tree for the wine dataset')
plt.show()
