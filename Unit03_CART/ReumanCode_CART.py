#Basic CART exercise for Wisconsin breast cancer dataset, in support of the CART
#unit of Biol 420/701: Machine Learning in Biology, University of Kansas.
#
#I use a version of the Wisconsin breast cancer dataset downloaded from here
#https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
#and then with some of the features stripped out for simplicity.

# %% **basic setup

import numpy as np
import pandas as pd
import math
import scipy as sp
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.use("TkAgg") #I dunno what this really does, I found it online and
#it helps with some problems I was having plotting from the pycharm console.
#Without this, running the plotting below through the console (specifically
#the plt.show() command) crashed not only pycharm but my whole virtual machine.
#Repeatedly. This helped. Anyway this is just IT stuff, not conceptual.

# %% **load the data and simplify

d=pd.read_csv("BreastCancerData.csv") #Assumes the data stored locally
print(type(d))
print(d.head)
print(d.shape)
print(len(d))
print(d.columns)
print(d.dtypes)

d=d.iloc[:,1:12] #keep only the "mean" columns
d.columns=["diagnosis","radius","texture","perimeter","area","smoothness",
           "compactness","concavity","concave_points","symmetry",
           "fractal_dimension"] #renamed for simplicity
print(d.shape)
print(d.columns)
print(type(d))

# %% **split into validation and testing data sets

#split
np.random.seed(101)
dper=d.iloc[np.random.permutation(len(d)),:]
print(dper.shape)
print(dper.columns)
dval=dper.iloc[0:(math.floor(.75*len(dper))),:]
dtest=dper.iloc[(math.floor(.75*len(dper))):(len(dper)+1),:]

#some consistency checks
print(len(dval)+len(dtest))
print(len(dper))

#python CART routines like to have the data separated into the
#predictors and the response
Xval=dval.iloc[:,1:11]
yval=dval.diagnosis

Xtest=dtest.iloc[:,1:11]
ytest=dtest.diagnosis

# %% **do a CART using some "pre-pruning", control parameters chosen to
#match as well as possible the default control parameters of
#rpart::rpart in R.

#fit a CART
m_d=tree.DecisionTreeClassifier(max_depth=30,min_samples_split=20,
                                min_samples_leaf=7,
                                min_impurity_decrease=0.01,
                                max_features=None,
                                random_state=101)
print(type(m_d))
m_d.fit(Xval,yval)

# %% examine it

#plot a picture of it
feature_names=dval.columns[1:12].to_list()
plt.figure()
tree.plot_tree(m_d,filled=True,feature_names=feature_names)
plt.show()
#Note, this is similar, but not the same as what I got in R using the
#defaults. I dunno why and it might be hard to figure out, so I won't
#try. The biggest difference is the last split is actually based on a
#different feature here compared to R! It's similar enough that I am not
#going to worry about it for the current pedagogical purposes, we can
#still illustrate the same ideas in python as we did in R. But this
#indicates that results can be dependent on
#detailed specifics of the algorithm used, which can differ by
#implementation and can be hard to work out from documentation
#alone unless one is quite careful! More careful than I have been here!
#These implementation differences don't matter HERE, where our goals are
#pedagogical, but they might matter in your research application, so
#watch out!

#textual display
m_d_text=tree.export_text(m_d,feature_names=feature_names)
print(m_d_text)

# %% see the errors it makes on the training data
m_d_pred=m_d.predict(Xval)
print(metrics.confusion_matrix(yval,m_d_pred)) #pretty similar but not identical to the R one
print(sum(m_d_pred!=yval)/len(yval)) #similar error to R
print(1-metrics.accuracy_score(yval,m_d_pred)) #a built-in way of getting the same result
print(1-m_d.score(Xval,yval)) #another way to get the same thing

# %% **Now do a full CART, all the way down (i.e., no stopping criteria), to
#work toward points about bias-variance, cross validation, overfitting, etc.

#fit it
m_f=tree.DecisionTreeClassifier(max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,
                                min_impurity_decrease=0,max_features=None,
                                random_state=101)
type(m_f)
m_f.fit(Xval,yval)

#examine it
plt.figure()
tree.plot_tree(m_f,filled=True)
plt.show()
#I did not try to compare this with the R one

m_f_text=tree.export_text(m_f,feature_names=dval.columns[1:12].to_list())
print(m_f_text)

# %% see the errors it makes on the training data
m_f_pred=m_f.predict(Xval)
print(metrics.confusion_matrix(yval,m_f_pred)) #perfect, as it should be, by construction
print(sum(m_f_pred!=yval)/len(yval))
print(1-metrics.accuracy_score(yval,m_f_pred)) #a built-in way of getting the same result
print(1-m_f.score(Xval,yval)) #another way to get the same thing

# %% **Some convenient python functions for assessing your trees

print(m_d.get_depth())
print(m_f.get_depth())
print(m_d.get_n_leaves())
print(m_f.get_n_leaves())
print(m_d.tree_.node_count)
print(m_f.tree_.node_count)
#So m_f is much more complex, as expected

# %% **now do a manual cross validation exercise

numgp=10
gp=np.tile(np.arange(0,numgp),math.ceil(len(dval)/numgp))
gp=gp[0:len(dval)]
xerrs_d=np.repeat(np.nan,numgp)
xerrs_f=np.repeat(np.nan,numgp)
m_d_s=m_d
m_f_s=m_f
for counter in np.arange(0,numgp):
    #fit the models on all the data except the one group
    m_d_s.fit(Xval[gp!=counter], yval[gp!=counter])
    m_f_s.fit(Xval[gp!=counter], yval[gp!=counter])

    #get predictions for the left out group and get error rates
    m_d_pred_s=m_d_s.predict(Xval[gp==counter])
    xerrs_d[counter]=sum(m_d_pred_s!=yval[gp==counter])/sum(gp==counter)
    m_f_pred_s=m_f_s.predict(Xval[gp==counter])
    xerrs_f[counter]=sum(m_f_pred_s!=yval[gp==counter])/sum(gp==counter)

print(xerrs_d.mean())
print(xerrs_f.mean())
#So the x-val error is slightly lower for the pre-pruned model (which
#is also much simpler), though basically the same. But if you have two models
#that are about equally good you take the simpler one. So the more complex
#model is overfitted in the sense that we don't need all that extra complexity,
#it basically gives us no additional out-of-sample predictive power.
#
#Note that both these error rates are higher than the
#corresponding values for the in-sample validation set, which is what
#you would expect.

# %% An automated way of getting about the same cross validation
aerrs_d=1-cross_val_score(m_d,Xval,yval,cv=10)
aerrs_f=1-cross_val_score(m_f,Xval,yval,cv=10)
print(aerrs_d.mean())
print(aerrs_f.mean())
#Pretty similar outcome to doing it by hand, above, though the numbers
#are slightly different, probably due to differences in the algorithm used
#which would be hard to track down. For instance, it could relate to how
#the specific groups were generated for the cross-validation. Or it could
#be that different metric of predictive error is being used, better
#than simple classification error.

# %% **Now do post-pruning based on cross validation

#This is just illustrative, since it is all on the training data.
#We need to do x-val, see below.
path_d=m_d.cost_complexity_pruning_path(Xval,yval)
print(path_d.ccp_alphas) #the last one is the trivial tree with one node
print(path_d.impurities)
plt.figure()
plt.plot(path_d.ccp_alphas[:-1],path_d.impurities[:-1],marker="o",drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("leaf impurity")
plt.title("m_d, training data")
plt.show()

#Again illustrative, see below for x-val error
path_f=m_f.cost_complexity_pruning_path(Xval,yval)
print(path_f.ccp_alphas)
print(path_f.impurities)
plt.figure()
plt.plot(path_f.ccp_alphas[:-1],path_f.impurities[:-1],marker="o",drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("leaf impurity")
plt.title("m_f, training data")
plt.show()

# %% Do cross validation for different complexity levels of tree
ccp_alphas=path_f.ccp_alphas[:-1]
cvss=np.repeat(np.nan,len(ccp_alphas))
cvs_ses=np.repeat(np.nan,len(ccp_alphas))
nodecounts=np.repeat(np.nan,len(ccp_alphas))
maxdepths=np.repeat(np.nan,len(ccp_alphas))
numleaves=np.repeat(np.nan,len(ccp_alphas))
for counter in np.arange(0,len(ccp_alphas)):
    ccp_alpha=ccp_alphas[counter]
    m_p=tree.DecisionTreeClassifier(max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,
                                min_impurity_decrease=0,max_features=None,
                                random_state=101,ccp_alpha=ccp_alpha).fit(Xval,yval)
    sc=cross_val_score(m_p, Xval, yval, cv=10)
    cvss[counter]=sc.mean()
    cvs_ses[counter]=sp.stats.sem(sc)
    nodecounts[counter]=m_p.tree_.node_count
    maxdepths[counter]=m_p.tree_.max_depth
    numleaves[counter]=m_p.get_n_leaves()

fig, ax = plt.subplots(4, 1)
ax[0].plot(ccp_alphas, nodecounts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[1].plot(ccp_alphas, maxdepths, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[2].plot(ccp_alphas,numleaves,marker="o",drawstyle="steps-post")
ax[2].set_xlabel("alpha")
ax[2].set_ylabel("number of leaves")
ax[3].plot(ccp_alphas, 1-cvss, marker="o")
for counter in np.arange(0,len(ccp_alphas)):
    ccp_alpha=ccp_alphas[counter]
    ax[3].plot(np.repeat(ccp_alpha,2),[1-cvss[counter]+cvs_ses[counter],1-cvss[counter]-cvs_ses[counter]],c="r")
dl=min(1-cvss+cvs_ses)
ax[3].plot([min(ccp_alphas),max(ccp_alphas)],np.repeat(dl,2),c="grey")
ax[3].set_xlabel("alpha")
ax[3].set_ylabel("x-val error")
fig.tight_layout()
plt.show()

#The above says the tree you want is the one with alpha about 0.01,
#i.e., you take the smallest error rate, add the se, and then take the
#largest alpha which is still less than that.
bestind=np.where(1-cvss<dl)[0].max()
bestalpha=ccp_alphas[bestind]
print(1-cvss[bestind]) #so a 7.9% x-val error for that tree, similar to the
#original m_d (actually a bit better but probably not meaningfully)
print(nodecounts[bestind])
print(maxdepths[bestind])
print(numleaves[bestind])

print(aerrs_d.mean())
print(m_d.tree_.node_count)
print(m_d.get_depth())
print(m_d.get_n_leaves())

#So this final model is a little better and just as simple as the
#original m_d, but probably not meaningfully better. Go ahead and
#take it to be the best model so far.

# %% **Next do bagging (bootstrap aggregation) and see if we can beat
#the best model we have so far

estimator=tree.DecisionTreeClassifier(max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,
                                min_impurity_decrease=0,max_features=None)
print(type(estimator))
m_bag=BaggingClassifier(estimator=estimator,n_estimators=500,oob_score=True,
                        n_jobs=6,random_state=202)
print(type(m_bag))
m_bag.fit(Xval,yval)
print(1-m_bag.score(Xval,yval)) #error rate is zero! Need to check if it is overfitted

#do x-val
bagerrs_d=1-cross_val_score(m_bag,Xval,yval,cv=10)
print(bagerrs_d.mean())
#So this is a bit better then the best model we had so far!

# %% **Now do random forests

m_rf=RandomForestClassifier(n_estimators=1000) #The defaults are the same
#as the values I used for m_f, which is appropriate.
m_rf.fit(Xval,yval)
print(1-m_rf.score(Xval,yval)) #error rate is zero! Need to check if it is overfitted.

#do x-val
rferrs_d=1-cross_val_score(m_rf,Xval,yval,cv=10)
print(rferrs_d.mean())
#So this is not quite as good as the bagging result. So we'll call the 
#best model so far the bagging one.

# %%  **now do adaptive boosting, including cross validation

stump=tree.DecisionTreeClassifier(max_depth=1,random_state=1)
m_ada=AdaBoostClassifier(estimator=stump,n_estimators=100)
m_ada.fit(Xval,yval)
print(1-m_ada.score(Xval,yval)) 

adaerrs_d=1-cross_val_score(m_ada,Xval,yval,cv=10)
print(adaerrs_d.mean()) #comparable to the bagging result, maybe slightly better

# %% **now do gradient boosting, using xgboost

#xgboost has its own native interface (that is the xgb.DMatrix and xgb.train
#route used in the R version of this file), but it also provides a
#scikit-learn-style interface, which is simpler and which lets us keep using
#cross_val_score exactly as we did for all the models above. See
#https://xgboost.readthedocs.io/en/stable/python/python_intro.html

#xgboost wants the response coded as 0s and 1s rather than as "B" and "M". This
#is the equivalent of the as.integer(d_val[,1])-1 line in the R version, and it
#codes B as 0 and M as 1, same as R does.
yval_num=(yval=="M").astype(int)

m_xgb=XGBClassifier(n_estimators=500,max_depth=2,learning_rate=.05,
                    objective="binary:logistic",n_jobs=2,random_state=101)
m_xgb.fit(Xval,yval_num)
print(1-m_xgb.score(Xval,yval_num))
#These are the same hyperparameters used in the R version. max_depth=2 keeps the
#individual trees weak, which is the whole point of boosting. learning_rate
#(called eta in older versions and in most write-ups) is the shrinkage applied to
#each tree's contribution before it is added in. n_estimators is the number of
#boosting rounds, and it has to go up when learning_rate goes down.

#do x-val
xgberrs_d=1-cross_val_score(m_xgb,Xval,yval_num,cv=10)
print(xgberrs_d.mean()) #a bit better than the adaptive boosting result

# %% **Now test THE ONE SINGLE BEST MODEL on your testing data

#reminders of x-val scores on some of the best models, to choose
print(bagerrs_d.mean())
print(rferrs_d.mean())
print(adaerrs_d.mean())
print(xgberrs_d.mean())

# %% final test of best model
ytest_num=(ytest=="M").astype(int)
print(1-m_xgb.score(Xtest,ytest_num)) #Error rate is really good! Even lower than
#the x-val error in the validation set, though that it is just lucky.

