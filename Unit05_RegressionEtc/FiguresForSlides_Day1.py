#Python version of FiguresForSlides_Day1.R, provided as a resource for the
#python people. Makes a couple of figures for the regression trees slides which
#are not part of the assignment solutions (those figures come from
#../Assignment/ReumanCode_RegressionTrees.py). The figures are conceptually the
#same as the R ones, not formatted identically. Run with the working directory
#set to this folder. Saved with a _py suffix so as not to overwrite the pdfs
#from the R version, which are the ones on the slides.
#
#AI-produced translation (Claude, 2026-10-13) of the R script, run end to end
#in the course python environment and checked against the R output.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

d=pd.read_csv("UnitCaret/AbaloneData.csv")
d.sex=d.sex.astype("category")

#**a one-predictor tree, to show that a regression tree is a step function
#whose steps are node means

#rpart's maxdepth=2 allows two levels of splits below the root, i.e., at most
#4 leaves; max_depth=2 is the same thing in scikit-learn. rpart would also
#stop early on its defaults (cp=0.01, minsplit=20), so match those too, with
#ccp_alpha=0.01*var(rings) playing the role of cp (see the regression trees
#assignment solution for why) and minbucket=round(20/3)=7.
X1=d[["shell_weight"]]
y=d.rings
m1=tree.DecisionTreeRegressor(max_depth=2,min_samples_split=20,
                              min_samples_leaf=7,ccp_alpha=0.01*y.var(),
                              random_state=101)
m1.fit(X1,y)
print(tree.export_text(m1,feature_names=["shell_weight"])) #the analog of
#printing an rpart object: the splits and the leaf means

plt.figure(figsize=(7,5))
plt.scatter(d.shell_weight,d.rings,s=2,c="grey")
x=np.arange(0,1.011,.001)
plt.plot(x,m1.predict(pd.DataFrame({"shell_weight":x})),c="red",lw=3)
plt.xlabel("Shell weight (g)")
plt.ylabel("Rings")
plt.savefig("StepFunction_py.pdf")
plt.show()

plt.figure(figsize=(7,5))
tree.plot_tree(m1,feature_names=["shell_weight"],filled=True,fontsize=9)
plt.savefig("StepFunctionTree_py.pdf")
plt.show()

#**the same thing with the full one-predictor tree, to show what "all the way
#down" looks like for regression

#cp=0 (no complexity penalty) and minsplit=2 (split any node with 2 or more
#cases) in rpart; ccp_alpha=0, min_samples_split=2, min_samples_leaf=1 here.
#This gives a few more leaves than rpart does (886 vs 838) because rpart also
#has a default maximum depth of 30 and scikit-learn has none. Same picture.
m1f=tree.DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1,
                               ccp_alpha=0,random_state=101)
m1f.fit(X1,y)
print("leaves in the full tree:",m1f.get_n_leaves())

plt.figure(figsize=(7,5))
plt.scatter(d.shell_weight,d.rings,s=2,c="grey")
plt.plot(x,m1f.predict(pd.DataFrame({"shell_weight":x})),c="red",lw=1)
plt.xlabel("Shell weight (g)")
plt.ylabel("Rings")
plt.savefig("StepFunctionFull_py.pdf")
plt.show()

#**the response and one predictor, just to show the data

plt.figure(figsize=(7,5))
plt.hist(d.rings,bins=np.arange(0,31))
plt.xlabel("Rings")
plt.ylabel("Frequency")
plt.savefig("RingsHist_py.pdf")
plt.show()
