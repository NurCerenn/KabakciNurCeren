#CLAUDE>> Lines tagged "#CLAUDE>>" were written by Claude (an AI); lines tagged
#CLAUDE>> "#DAN>>" were written by Dan. These are learning suggestions only.
# This is my work for the assingment. I tried working with different datasets
# including antimicrobial peptides, anticancer peptides classification
# but i realized i dont have suitable dataset so decided going with Diabeties dataset

# DAN>> I am sorry the other ones did not work out! Usually, fnding/building a 
# DAN>> suitable dataset is, indeed, the hardest part! If you were trying to use
# DAN>> to use one from your lab then you were undertaking a real challenge for 
# DAN>> such a short-timeline assignment!

# After long trial and error with different datasets
# here will try to classify Diabeties risk of patient based on the sign and symptpom data 
# of newly diabetic or would be diabetic patient. Here I go.

#set-up 
import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn import metrics

# let's look at the data
d=pd.read_csv("diabetes_data_upload.csv")
print(type(d))
print(d.head)
print(d.shape)
print(d.dtypes)
print(len(d))
print(d.info())
print(d.columns)


# I need label encoding for the dataset I'm using
from sklearn.preprocessing import LabelEncoder
label_cols=['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'diabetes']
d[label_cols]=d[label_cols].apply(LabelEncoder().fit_transform)
print(type(d))



# manual training and test seperation
np.random.seed(101) 
d_per=d.iloc[np.random.permutation(len(d)),:]
d_train=d_per.iloc[0:(math.floor(.75*len(d_per)))]
d_test=d_per.iloc[(math.floor(.75*len(d_per))):(len(d_per)+1):]

# checking if the tes and training split was sucsessful 
print(d_per.columns)
print(len(d_train))
print(len(d_test))
print(len(d_per))

print(d_train.shape)
print(d_test.shape)
print(d_per.shape)

# looks like it was, the numbers are consistent, I can move forward 
# to work on my CART


# first, defining the predictors
#CLAUDE>> ISSUE: iloc[:,0:15] takes columns 0-14, which is 15 columns, so Obesity (the 16th feature) is
#CLAUDE>> left out of every model. iloc[:,0:16] includes all 16. Same on the Xtest line below.
# DAN>> I agree with Claude. It'll be worth it for you to review how python does indexing - it can be non-intuitive at first. 
Xtrain=d_train.iloc[:,0:15]  #first 16 colunms are the features
ytrain=d_train.diabetes #'class' is the diabeties, what we will classify

Xtest=d_test.iloc[:,0:15]
ytest=d_test.diabetes 

# time to fit the tree

from sklearn import tree
mytree_d=tree.DecisionTreeClassifier(max_depth=30,max_features=None,
                                     min_samples_leaf=7,min_impurity_decrease=0.01,
                                     random_state=101,min_samples_split=20)
mytree_d.fit(Xtrain,ytrain)

# let's see what we got now
print(tree.plot_tree(mytree_d)) #this gave me the a text result for my tree

import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import scipy as sp
feature_names=d_train.columns[0:16].to_list()
plt.figure()
tree.plot_tree(mytree_d,filled=True,feature_names=feature_names)
plt.show()


# let's do the the prediction 
mytree_d_pred=mytree_d.predict(Xtrain)

#what about the errors?
print(metrics.confusion_matrix(ytrain,mytree_d_pred)) 
#[[127  22]
#[ 14 227]]
print(sum(mytree_d_pred!=ytrain)/len(ytrain)) #0.09230769230769231
print(1-metrics.accuracy_score(ytrain,mytree_d_pred)) #0.09230769230769231

# time to do a full CART to see if there are any difference
mytree_f=tree.DecisionTreeClassifier(max_depth=None,max_features=None,
                                     min_samples_leaf=1,min_impurity_decrease=0,
                                     random_state=101,min_samples_split=2)
mytree_f.fit(Xtrain,ytrain)
plt.figure()
tree.plot_tree(mytree_f,filled=True,feature_names=feature_names)
plt.show() #got a huge tree this time

mytree_f_pred=mytree_f.predict(Xtrain)

print(metrics.confusion_matrix(ytrain,mytree_f_pred))
#[[149   0]
#[  0 241]]
print(sum(mytree_f_pred!=ytrain)/len(ytrain))  #0.0 overfitted
print(1-metrics.accuracy_score(ytrain,mytree_f_pred))  #0.0

# manual cross validation as ve practiced in the class

numgp=10 #number of groups
gp=np.tile(np.arange(0,numgp),math.ceil(len(d_train)/numgp))
gp=gp[0:len(d_train)]
xerrs_d=np.repeat(np.nan,numgp) #error of restricted tree
xerrs_f=np.repeat(np.nan,numgp) #error of full tree

#CLAUDE>> NOTE: in Python this makes a second NAME for the same object, so the loop refits mytree_d itself.
#CLAUDE>> from sklearn.base import clone; mytree_d_s=clone(mytree_d) gives an independent copy.
mytree_d_s=mytree_d
mytree_f_s=mytree_f # we do this so that it doesn't overrides original trees
# DAN>> Claude is right. This was actually an error in my code (since fixed). For a mutable object 
# DAN>> x, the assignment y = x actually just creates a new *name*, y, which points to the same 
# DAN>> underlying object as x. Any modifications to y will affect x as well. Do this instead:
# DAN>> from sklearn.base import clone
# DAN>> mytree_d_s=clone(mytree_d)
# DAN>> mytree_f_s=clone(mytree_f)
# DAN>> That make so-called "deep copies"

for counter in np.arange(0,numgp):
    mytree_d_s.fit(Xtrain[gp!=counter], ytrain[gp!=counter])
    mytree_f_s.fit(Xtrain[gp!=counter], ytrain[gp!=counter])

    mytree_d_pred_s=mytree_d_s.predict(Xtrain[gp==counter])
    xerrs_d[counter]=sum(mytree_d_pred_s!=ytrain[gp==counter])/sum(gp==counter)
    mytree_f_pred_s=mytree_f_s.predict(Xtrain[gp==counter])
    xerrs_f[counter]=sum(mytree_f_pred_s!=ytrain[gp==counter])/sum(gp==counter)

#CLAUDE>> THINK: your full tree's CV error (0.041) is lower than the default's (0.131), the reverse of the
#CLAUDE>> class breast-cancer result. Day 3 asks you to explain this in comments: why might a deep tree
#CLAUDE>> generalize well on 16 yes/no symptom columns?
print(xerrs_d.mean())  #0.1307692307692308
print(xerrs_f.mean())  #0.041025641025641026

#let's see the error from automated way
from sklearn.model_selection import cross_val_score
aerrs_d=1-cross_val_score(mytree_d,Xtrain,ytrain,cv=10)
aerrs_f=1-cross_val_score(mytree_f,Xtrain,ytrain,cv=10)

print(aerrs_d.mean())   #0.11282051282051282
print(aerrs_f.mean())   #0.03333333333333334

#more or so similar with my manual cross validation, not bad

#now, time for some post-pruning, I studied the script ant tried my best to write
#it all by my own but there were times I needed help so I peeked to the class notes :/

path_d=mytree_d.cost_complexity_pruning_path(Xtrain,ytrain)
print(path_d.ccp_alphas)
print(path_d.impurities)

plt.figure
plt.plot(path_d.ccp_alphas[:-1],path_d.impurities[:-1],marker='o')
plt.xlabel("effective alpha")
plt.ylabel("leaf impurity")
plt.title("mytree_d, training data")
plt.show()


path_f=mytree_f.cost_complexity_pruning_path(Xtrain,ytrain)
print(path_f.ccp_alphas)
print(path_f.impurities)

plt.figure
plt.plot(path_f.ccp_alphas[:-1],path_f.impurities[:-1],marker='o')
plt.xlabel("effective alpha")
plt.ylabel("leaf impurity")
plt.title("mytree_f, training data")
plt.show()

# I wish there was an easier way to do this in py just like R
# time for cross validation for different trees with different complexty so that we can pick one
import scipy.stats as stats 

ccp_alphas=path_f.ccp_alphas[:-1]
cvss=np.repeat(np.nan,len(ccp_alphas))
cvs_ses=np.repeat(np.nan,len(ccp_alphas))
nodecounts=np.repeat(np.nan,len(ccp_alphas))
maxdepths=np.repeat(np.nan,len(ccp_alphas))
numleaves=np.repeat(np.nan,len(ccp_alphas))
for counter in np.arange(0,len(ccp_alphas)):
    ccp_alpha=ccp_alphas[counter]
    mytree_p=tree.DecisionTreeClassifier(max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,
                                min_impurity_decrease=0,max_features=None,
                                random_state=101,ccp_alpha=ccp_alpha).fit(Xtrain,ytrain) 
    sc=cross_val_score(mytree_p, Xtrain, ytrain, cv=10)
    cvss[counter]=sc.mean()
    cvs_ses[counter] = stats.sem(sc)
    nodecounts[counter]=mytree_p.tree_.node_count
    maxdepths[counter]=mytree_p.tree_.max_depth
    numleaves[counter]=mytree_p.get_n_leaves()

fig, ax =plt.subplots(4,1)
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

#geez this was a lovely challange to get used to coding visuals and get the syntax right
# DAN>> Yes, plotting is usually a pain in the neck in any language

for counter in np.arange(0,len(ccp_alphas)):
   ccp_alpha=ccp_alphas[counter]
   ax[3].plot(np.repeat(ccp_alpha,2),[1-cvss[counter]+cvs_ses[counter],
                                      1-cvss[counter]-cvs_ses[counter]],c="r")
   dl=min(1-cvss+cvs_ses)
ax[3].plot([min(ccp_alphas),max(ccp_alphas)],np.repeat(dl,2),c="pink")
ax[3].set_xlabel("alpha")
ax[3].set_ylabel("x-val error")
fig.tight_layout()
plt.show()

# 0.006 

#CLAUDE>> GOOD: taking the largest alpha whose CV error is under the min + SE line picks the simplest
#CLAUDE>> adequate tree. That is the 1-SE rule, and your plot shows the SE bars it relies on.
bestind=np.where(1-cvss<dl)[0].max()
bestalpha=ccp_alphas[bestind]
print(1-cvss[bestind]) #0.03333333333333344
print(nodecounts[bestind])  #51.0
print(maxdepths[bestind])  #8.0
print(numleaves[bestind])  #26.0

print(aerrs_d.mean()) #0.11282051282051282
print(mytree_d.tree_.node_count)  #13
print(mytree_d.get_depth())  #4
print(mytree_d.get_n_leaves())  #7


# its bagging time, I was absent in this class, wish me luck
# DAN>> You can always watch the recording, and you are encouraged to! That's why I make them!
from sklearn.ensemble import BaggingClassifier

estimator=tree.DecisionTreeClassifier(max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,
                                min_impurity_decrease=0,max_features=None)
print(type(estimator))
m_bag=BaggingClassifier(estimator=estimator,n_estimators=500,oob_score=True,
                        n_jobs=6,random_state=202)
print(type(m_bag))
m_bag.fit(Xtrain,ytrain)
#CLAUDE>> NOTE: 0 within-sample is expected here, since every bagged tree is full and memorizes its sample.
#CLAUDE>> Your CV error just below (0.031) is the number that tells you how it does on new data.
print(1-m_bag.score(Xtrain,ytrain))  # error rate is 0.0, overfitted?

bagerrs_d=1-cross_val_score(m_bag,Xtrain,ytrain,cv=10)
print(bagerrs_d.mean())  #0.03076923076923076

#this is a slightly better model by 0.003

# let's see how random forest will perform, I love RFs for no professional reasons
from sklearn.ensemble import RandomForestClassifier

my_rf=RandomForestClassifier(n_estimators=1000)
my_rf.fit(Xtrain,ytrain)
print(1-my_rf.score(Xtrain,ytrain)) #0.0

#Xtrain to check if overfits
rferrs_d=1-cross_val_score(my_rf,Xtrain,ytrain,cv=10)
print(rferrs_d.mean()) #0.015384615384615396

# wow RF did better, RF <3 me
# DAN>> And your error is very low!

#now lets see adaptive boosting
from sklearn.ensemble import AdaBoostClassifier

stump=tree.DecisionTreeClassifier(max_depth=1,random_state=1)
#CLAUDE>> THINK: Day 6 asks you to try a few settings (n_estimators, learning_rate, stump depth) and let
#CLAUDE>> cross_val_score decide. Does 100 stumps beat, say, 300?
my_ada=AdaBoostClassifier(estimator=stump,n_estimators=100)
my_ada.fit(Xtrain,ytrain)

print(1-my_ada.score(Xtrain,ytrain)) #0.06153846153846154
adaerrs_d=1-cross_val_score(my_ada,Xtrain,ytrain,cv=10)
print(adaerrs_d.mean()) #0.07435897435897434

#adaptive boosting wasn't the best one so far, if not the worst one

# one last model, gradient boosting with xgboost
from xgboost import XGBClassifier

#CLAUDE>> ISSUE: your labels were already made 0/1 by LabelEncoder (line 33), so ytrain=='M' is False
#CLAUDE>> everywhere and ytrain_num is all zeros. XGBoost learns "always 0", which is why its error
#CLAUDE>> shows 0.0. That isn't overfitting. Pass ytrain directly; it is already numeric.
# DAN>>  I think this is the error which led to the apparent perfecet accuracy you repoted during your presentation!
ytrain_num=(ytrain=='M').astype(int)

my_xgb=XGBClassifier(n_estimators=500, max_depth=2,learning_rate=.05,
                    objective="binary:logistic",n_jobs=2,random_state=101)
my_xgb.fit(Xtrain,ytrain_num)

print(1-my_xgb.score(Xtrain,ytrain_num)) #0.0

#do x-val
xgberrs_d=1-cross_val_score(my_xgb,Xtrain,ytrain_num,cv=10) #error is 0.0 I guess this model was overfitted

#let's recall all the x-val scores of the models, to choose the best one
print(bagerrs_d.mean())
print(rferrs_d.mean())
print(adaerrs_d.mean())
print(xgberrs_d.mean())

# last but the not least, lets use our test dataset with our best model, RF

#CLAUDE>> ISSUE: my_rf.fit(Xtest,ytest) trains the forest ON the test set, so 0.0 is it scoring its own
#CLAUDE>> homework, and cross_val_score(...Xtest...) is CV inside the vault. For Day 7, fit on Xtrain/ytrain
#CLAUDE>> (as at line 258) and call my_rf.predict(Xtest) once.
my_rf=RandomForestClassifier(n_estimators=1000)
my_rf.fit(Xtest,ytest)
# DAN>> Agree with Claude - here you are re-training the RF on the test set!
print(1-my_rf.score(Xtest,ytest)) #0.0

# I'm crossing my fingers as I hit the run button
rferrs_t=1-cross_val_score(my_rf,Xtest,ytest,cv=10)
print(rferrs_t.mean())  #0.04615384615384614

# fiyuu, this wasn't so bad, actually pretty good I guess.
#lets predict
my_rf_pred=my_rf.predict(Xtest)
from sklearn.metrics import confusion_matrix
print(metrics.confusion_matrix(ytest,my_rf_pred))
print(1-metrics.accuracy_score(ytest,my_rf_pred))
print(metrics.classification_report(ytest,my_rf_pred))

from sklearn.metrics import ConfusionMatrixDisplay
cmatrix_my_rf=confusion_matrix(ytest,my_rf_pred)
disp_cmatrix_my_rf=ConfusionMatrixDisplay(confusion_matrix=cmatrix_my_rf)
disp_cmatrix_my_rf.plot()
print(disp_cmatrix_my_rf)


#CLAUDE>> NOTE: from_predictions expects (y_true, y_pred), so swap these two. And plt.show two lines down
#CLAUDE>> has no (), so it is never called. That is probably why no figure appeared.
ConfusionMatrixDisplay.from_predictions(my_rf_pred, ytest, cmap='Blues')
plt.title("RF Confusion Matrix Test Data")
plt.show 

#tried to get confusion matrix figure but I wasnt successfull :(