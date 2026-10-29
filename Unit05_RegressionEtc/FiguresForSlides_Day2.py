#Makes the printed outputs shown on the PyCaret slides of the Day 2 (caret)
#lecture. Run with the working directory set to this folder, in an environment
#with PyCaret installed (python 3.9-3.11; see the "Installing PyCaret" section
#of the assignment). Uses the abalone data and the same validation/testing split
#as Day 1. Output is also saved to SlideOutputs_py.txt.

# %% setup

import sys
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from pycaret.regression import RegressionExperiment
#object style: an explicit experiment object, with every PyCaret call a method
#of it; no hidden global experiment
pd.set_option("display.width",120)
pd.set_option("display.max_columns",20)

# %% **the data and the split, as on Day 1

d=pd.read_csv("AbaloneData.csv")
d.sex=d.sex.astype("category")
np.random.seed(101)
dper=d.iloc[np.random.permutation(len(d)),:]
dval=dper.iloc[0:(math.floor(.75*len(dper))),:]
dtest=dper.iloc[(math.floor(.75*len(dper))):(len(dper)+1),:]
print(dval.shape); print(dtest.shape)

# %% **setup an "experiment"

print("\n*** setup")
exp=RegressionExperiment()
exp.setup(data=dval,target="rings",test_data=dtest,
          fold_strategy=KFold(n_splits=10,shuffle=True,random_state=102),
          normalize=True,session_id=103,n_jobs=1)
#This specifies the the validation and testing data, the response 
#variable, the folds, whether variables are normalized, the random seed,
#and number cores. These apply to all downstream models.

# %% **look at available models that PyCaret provides

print("\n*** models")
ml=exp.models()
print(ml.shape)
print(ml[["Name"]])

# %% **create_model and tune_model, dt

print("\n*** create_model dt")
m_dt=exp.create_model("dt") #just does x-val using a decision tree with
#the default tuning parameters
print("\n*** tune_model dt")
m_dt2=exp.tune_model(m_dt,n_iter=25,optimize="RMSE") #explores 25 sets 
#of tuning parameters, gives you x-val results on the best
print(m_dt2)

# %% **knn

print("\n*** create_model knn")
m_knn=exp.create_model("knn")
print("\n*** tune_model knn with a custom grid")
m_knn2=exp.tune_model(m_knn,custom_grid={"n_neighbors":list(range(5,34,2))},
                  search_algorithm="grid",optimize="RMSE")
print(m_knn2)
print(exp.pull().loc[["Mean","Std"]])

# %% **compare_models
print("\n*** compare_models")
best=exp.compare_models(include=["lr","dt","knn","svm","rf"],sort="RMSE")
print(best)

# %% **predict_model

print("\n*** predict_model")
out=exp.predict_model(m_knn2,data=dval)
print(out[["rings","prediction_label"]].head())
