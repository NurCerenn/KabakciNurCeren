#Python version of FiguresForSlides_Day3.R, provided as a resource for the
#python people. Makes the figures and the printed outputs shown on the Day 3
#(SVMs, logistic regression, regularization) slides, using scikit-learn where
#the R uses caret, kernlab and glmnet. Run with the working directory set to
#this folder. Uses the breast cancer data and the abalone data with the same
#kind of validation/testing split as the R, but numpy's random permutation is
#not R's sample(), so the split and the folds differ and the numbers differ a
#bit from the slides. Same overall conclusions. Figures are saved with a _py 
#suffix. Printed output is also saved to SlideOutputs_py.txt.
#
#Note: This is a fairly close, but not perfect parallel of the R code. 
#Differences come from various conventional differences between the R and python
#packages this script relies on. 
#
#The main differences from the R, each noted where it comes up: scikit-learn
#has no modelLookup, so the parameter grids are written out; caret's
#"preProcess" is a StandardScaler at the front of a Pipeline, so the scaling
#is re-estimated inside each fold as caret does; caret's Accuracy, Kappa and
#AccuracySD columns are rebuilt from GridSearchCV's cv_results_; and
#scikit-learn's regularization parameters are not glmnet's, so the conversions
#are written out.
#
#AI-produced translation (Claude, 2026-10-20) of the R script, run end to end
#in the course python environment and checked against the R output.

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression
from sklearn.datasets import load_iris

pd.set_option("display.width",120)
pd.set_option("display.max_columns",20)

#the analog of sink(split=TRUE): everything printed also goes to a file
class Tee:
    def __init__(self,fname):
        self.file=open(fname,"w"); self.stdout=sys.stdout
    def write(self,s):
        self.file.write(s); self.stdout.write(s)
    def flush(self):
        self.file.flush(); self.stdout.flush()
sys.stdout=Tee("SlideOutputs_py.txt")

#caret reports Accuracy, Kappa and their SDs across folds for every point on
#the grid; GridSearchCV stores the same things in cv_results_ under other
#names. This pulls them out in caret's layout, i.e., the analog of m$results.
scoring_class={"Accuracy":"accuracy","Kappa":make_scorer(cohen_kappa_score)}
def results_table(gs):
    cvr=pd.DataFrame(gs.cv_results_)
    out=pd.DataFrame(list(cvr["params"]))
    out["Accuracy"]=cvr["mean_test_Accuracy"]
    out["Kappa"]=cvr["mean_test_Kappa"]
    out["AccuracySD"]=cvr["std_test_Accuracy"]*math.sqrt(10/9) #sklearn's std is
    #the population one; caret's sd is the sample one, so rescale for 10 folds
    return out

#caret's grid for glmnet: alpha in seq(0.1,1,length=tuneLength), and lambdas
#taken from glmnet's own path, which runs down from the smallest lambda that
#zeros every coefficient (lambda_max) in equal log steps, with the two ends of
#a 7-point path dropped to leave 5. This rebuilds that grid, so the lambdas
#are on glmnet's scale and comparable to the slides. y should be the response
#(0/1 for logistic), X the predictors before scaling.
def caret_lambda_grid(X,y,tuneLength=5):
    Xs=(X-X.mean())/X.std(ddof=0) #glmnet standardizes with the population sd
    lambda_max=np.max(np.abs(Xs.T@(y-np.mean(y))))/len(y)/0.5 #/alpha, with alpha=.5 as caret uses
    path=lambda_max*10**(-4*np.arange(tuneLength+2)/(tuneLength+1)) #glmnet: down to 1e-4*lambda_max
    return path[1:-1]

#**the breast cancer data and split, as in the CART unit

dbc=pd.read_csv("BreastCancerData.csv")
dbc=dbc.iloc[:,1:12] #keep only the "mean" columns
dbc.columns=["diagnosis","radius","texture","perimeter","area","smoothness",
             "compactness","concavity","concave_points","symmetry","fractal_dimension"]
dbc.diagnosis=dbc.diagnosis.astype("category")
np.random.seed(101)
dbc_perm=dbc.iloc[np.random.permutation(len(dbc)),:]
dbc_val=dbc_perm.iloc[0:math.floor(.75*len(dbc)),:]
dbc_test=dbc_perm.iloc[math.floor(.75*len(dbc)):len(dbc),:]
print(dbc_val.shape)
print(dbc_test.shape)
Xbc=dbc_val.drop(columns="diagnosis") #scikit-learn wants X and y separately,
ybc=dbc_val.diagnosis                 #where caret takes a formula

#**the abalone data and split, as on Days 1 and 2

dab=pd.read_csv("AbaloneData.csv")
dab.sex=dab.sex.astype("category")
np.random.seed(101)
dab_perm=dab.iloc[np.random.permutation(len(dab)),:]
dab_val=dab_perm.iloc[0:math.floor(.75*len(dab)),:]
dab_test=dab_perm.iloc[math.floor(.75*len(dab)):len(dab),:]
print(dab_val.shape)
print(dab_test.shape)
#R's formula interface turns sex into two 0/1 dummy columns automatically;
#here it has to be done by hand
Xab=pd.get_dummies(dab_val.drop(columns="rings"),columns=["sex"],drop_first=True,dtype=float)
yab=dab_val.rings

#**SVMs in caret on the breast cancer data: svmLinear, fixed folds

print("\n*** svmLinear on breast cancer, fixed folds")
#no modelLookup in scikit-learn; SVC(kernel="linear") has the one tuning
#parameter C. createFolds on a factor stratifies, so StratifiedKFold here.
#Materializing the folds as a list is the analog of trainControl(index=...):
#the same folds get reused for every model below.
foldsbc=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=102).split(Xbc,ybc))
m_svmL=GridSearchCV(Pipeline([("scale",StandardScaler()),("m",SVC(kernel="linear"))]),
                    param_grid={"m__C":2.0**np.arange(-4,5)},cv=foldsbc,
                    scoring=scoring_class,refit="Accuracy")
m_svmL.fit(Xbc,ybc)
print(results_table(m_svmL))
print(results_table(m_svmL).Kappa.max())
print(m_svmL.best_params_)

#**SVMs: the effect of sigma and C for svmRadial, on two predictors, to show
#what the two tuning parameters do

#two predictors, centered and scaled, for drawing pictures
x2=dbc_val[["radius","texture"]]
x2=((x2-x2.mean())/x2.std()).to_numpy()
y2=dbc_val.diagnosis
gx=np.linspace(-3,4,150)
gy=np.linspace(-3,4,150)
GX,GY=np.meshgrid(gx,gy)
grid2=np.column_stack([GX.ravel(),GY.ravel()])

sigmas=[0.1,10]
Cs=[0.1,100]
fig,axs=plt.subplots(2,2,figsize=(7,7))
for i,sg in enumerate(sigmas):
    for j,C in enumerate(Cs):
        m=SVC(kernel="rbf",gamma=sg,C=C) #scikit-learn's gamma is kernlab's
        #sigma, exactly: both kernels are exp(-gamma*||x-x'||^2)
        m.fit(x2,y2)
        z=m.decision_function(grid2) #this is the signed distance to the
        #hyperplane, so 0 is the decision boundary and +/-1 are the gutters
        acc=np.mean(m.predict(x2)==y2)
        ax=axs[i,j]
        ax.scatter(x2[:,0],x2[:,1],c=np.where(y2=="M","red","blue"),s=4)
        ax.contour(GX,GY,z.reshape(GX.shape),levels=[-1,0,1],colors="black",
                   linewidths=[1,2,1],linestyles=["dashed","solid","dashed"])
        ax.set_xlim(gx.min(),gx.max()); ax.set_ylim(gy.min(),gy.max())
        ax.set_xlabel("radius (scaled)"); ax.set_ylabel("texture (scaled)")
        ax.set_title(f"sigma={sg}, C={C}, train acc {acc:.3f}",fontsize=10)
fig.tight_layout()
fig.savefig("SvmRadialGrid_py.pdf")
plt.close(fig)

#**svmRadial in caret on the breast cancer data, same folds, here for completeness
#*even though we did it already in an eaarlier class session

print("\n*** svmRadial on breast cancer, fixed folds")
#caret's default grid for svmRadial fixes sigma at a data-based estimate (from
#kernlab's sigest) and varies C over 0.25,0.5,1,2,4. The closest scikit-learn
#analog is gamma="scale", which is 1/(number of predictors) on scaled data,
#with the same C grid.
m_svmR=GridSearchCV(Pipeline([("scale",StandardScaler()),("m",SVC(kernel="rbf",gamma="scale"))]),
                    param_grid={"m__C":[0.25,0.5,1,2,4]},cv=foldsbc,
                    scoring=scoring_class,refit="Accuracy")
m_svmR.fit(Xbc,ybc)
print(results_table(m_svmR))
print(results_table(m_svmR).Kappa.max())
print(m_svmR.best_params_)

#**logistic regression: the expit, and a picture in 2D

print("\n*** logistic regression on two predictors")
#statsmodels gives the glm-style coefficient table with standard errors;
#scikit-learn's LogisticRegression does not
m_lr2=sm.GLM(ybc=="M",sm.add_constant(dbc_val[["radius","texture"]]),
             family=sm.families.Binomial()).fit()
print(m_lr2.summary().tables[1])
fig,axs=plt.subplots(1,2,figsize=(9,4))
u=np.arange(-6,6.01,.01)
axs[0].plot(u,1/(1+np.exp(-u)),"k-",lw=2)
axs[0].axhline(0,ls="--",c="k"); axs[0].axhline(1,ls="--",c="k")
axs[0].axhline(.5,ls=":",c="k"); axs[0].axvline(0,ls=":",c="k")
axs[0].set_xlabel("w.x+b"); axs[0].set_ylabel("p = expit(w.x+b)")
axs[0].set_title("The expit (logistic) function")
gxr=np.linspace(5,30,150)
gyt=np.linspace(8,40,150)
GR,GT=np.meshgrid(gxr,gyt)
gr=pd.DataFrame({"radius":GR.ravel(),"texture":GT.ravel()})
p=m_lr2.predict(sm.add_constant(gr))
axs[1].scatter(dbc_val.radius,dbc_val.texture,c=np.where(dbc_val.diagnosis=="M","red","blue"),s=4)
cs=axs[1].contour(GR,GT,np.asarray(p).reshape(GR.shape),levels=[.1,.5,.9],colors="black",
                  linewidths=[1,2,1],linestyles=["dashed","solid","dashed"])
axs[1].clabel(cs)
axs[1].set_xlabel("radius"); axs[1].set_ylabel("texture")
axs[1].set_title("Breast cancer: p(M), two predictors")
fig.tight_layout()
fig.savefig("LogisticRegression_py.pdf")
plt.close(fig)

#**logistic regression in caret, with all predictors

print("\n*** logistic regression in caret")
#no tuning parameters, so cross_validate instead of GridSearchCV. IMPORTANT:
#scikit-learn's LogisticRegression regularizes by default (ridge-style, C=1);
#C=np.inf turns that off and makes it ordinary logistic regression like glm.
pipe_glm=Pipeline([("scale",StandardScaler()),("m",LogisticRegression(C=np.inf,max_iter=5000))])
cv_glm=cross_validate(pipe_glm,Xbc,ybc,cv=foldsbc,scoring=scoring_class)
print(pd.DataFrame({"Accuracy":[cv_glm["test_Accuracy"].mean()],
                    "Kappa":[cv_glm["test_Kappa"].mean()],
                    "AccuracySD":[cv_glm["test_Accuracy"].std(ddof=1)],
                    "KappaSD":[cv_glm["test_Kappa"].std(ddof=1)]}))
m_glm=pipe_glm.fit(Xbc,ybc) #caret refits on all the data for finalModel; by hand here
print(pd.Series(np.concatenate([m_glm["m"].intercept_,m_glm["m"].coef_[0]]),
                index=["(Intercept)"]+list(Xbc.columns)).round(2)) #crazy radius and area coefs
ptr=pd.DataFrame(m_glm.predict_proba(Xbc),columns=m_glm.classes_,index=Xbc.index)
print(ptr.head(3)) #just to show you get useful probabilistic predictions

#**glmnet in caret on abalone, same folds as Day 2

print("\n*** glmnet in caret on abalone, same folds as Day 2")
#ElasticNet is glmnet with the names changed: scikit-learn's l1_ratio is
#glmnet's alpha, and scikit-learn's alpha is glmnet's lambda. caret's
#tuneLength=5 gives alpha in 0.1,0.325,0.55,0.775,1 and five lambdas taken
#from glmnet's own path; caret_lambda_grid (top of file) rebuilds the latter.
foldsab=list(KFold(n_splits=10,shuffle=True,random_state=102).split(Xab,yab))
m_glmnet=GridSearchCV(Pipeline([("scale",StandardScaler()),("m",ElasticNet(max_iter=10000))]),
                      param_grid={"m__l1_ratio":np.linspace(0.1,1,5),
                                  "m__alpha":caret_lambda_grid(Xab,yab)},cv=foldsab,
                      scoring={"RMSE":"neg_root_mean_squared_error","Rsquared":"r2"},refit="RMSE")
m_glmnet.fit(Xab,yab)
res=pd.DataFrame(m_glmnet.cv_results_)
res=pd.DataFrame({"alpha":res["param_m__l1_ratio"],"lambda":res["param_m__alpha"],
                  "RMSE":-res["mean_test_RMSE"],"Rsquared":res["mean_test_Rsquared"]})
print(res)
print(m_glmnet.best_params_) #very small lambda means basically we are doing ordinary regression
print(res.RMSE.min())
cv_lm=cross_validate(Pipeline([("scale",StandardScaler()),("m",LinearRegression())]),Xab,yab,
                     cv=foldsab,scoring={"RMSE":"neg_root_mean_squared_error","Rsquared":"r2"})
print(pd.DataFrame({"RMSE":[-cv_lm["test_RMSE"].mean()],
                    "Rsquared":[cv_lm["test_Rsquared"].mean()]})) #so regression does almost the same on RMSE

#**glmnet in caret on breast cancer data, same folds as Day 2. glmnet uses
#*logistic regression, here, because the response is categorical.

print("\n*** glmnet in caret on breast cancer, same folds as Day 2")
#LogisticRegression with an l1_ratio between 0 and 1 is the elastic net, i.e.,
#glmnet's binomial family, but its C is 1/(n*lambda), where n is the number of
#training rows, and l1_ratio is glmnet's alpha. So lambda is converted to C for
#the grid and back for the table. saga is the only solver that does the elastic
#net penalty.
lambdas=caret_lambda_grid(Xbc,(ybc=="M").astype(float))
n_bc=len(ybc)
m_glmnet_bc=GridSearchCV(Pipeline([("scale",StandardScaler()),
                                   ("m",LogisticRegression(solver="saga",max_iter=20000))]),
                         param_grid={"m__l1_ratio":np.linspace(0.1,1,5),"m__C":1/(n_bc*lambdas)},
                         cv=foldsbc,scoring=scoring_class,refit="Accuracy")
m_glmnet_bc.fit(Xbc,ybc)
res_bc=results_table(m_glmnet_bc)
res_bc=pd.DataFrame({"alpha":res_bc["m__l1_ratio"],"lambda":1/(n_bc*res_bc["m__C"]),
                     "Accuracy":res_bc.Accuracy,"Kappa":res_bc.Kappa})
print(res_bc)
print({"alpha":m_glmnet_bc.best_params_["m__l1_ratio"],
       "lambda":1/(n_bc*m_glmnet_bc.best_params_["m__C"])}) #on the R split on
#the slides, alpha=1 won, i.e., the LASSO. On this split several alphas tie at
#the best lambda and GridSearchCV reports the first, alpha=0.1, mostly ridge.
#Same lambda as on the slides, near enough.
fm=m_glmnet_bc.best_estimator_["m"] #best_estimator_ is caret's finalModel: refit on all the data
print(pd.Series(np.concatenate([fm.intercept_,fm.coef_[0]]),
                index=["(Intercept)"]+list(Xbc.columns)).round(2)) #compare the glm coefs: now
#interpretable. Mostly ridge, so nothing is exactly 0 here; the LASSO fit on
#the slides zeroed four. Not in the R: the LASSO at the same lambda, to show that
fm1=clone(m_glmnet_bc.best_estimator_).set_params(m__l1_ratio=1).fit(Xbc,ybc)["m"]
print(pd.Series(np.concatenate([fm1.intercept_,fm1.coef_[0]]),
                index=["(Intercept)"]+list(Xbc.columns)).round(2)) #some exactly 0
print(res_bc.Accuracy.max(), cv_glm["test_Accuracy"].mean()) #accuracy about the same as glm, but now coefficients are interpretable

#**multi-class: a binary method on three classes, caret just does it

print("\n*** multi-class, iris")
iris=load_iris(as_frame=True)
Xir=iris.data
yir=iris.target_names[iris.target]
print(pd.Series(yir).value_counts())
foldsir=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=103).split(Xir,yir))
cv_iris=cross_validate(Pipeline([("scale",StandardScaler()),("m",SVC(kernel="linear",C=1))]),
                       Xir,yir,cv=foldsir,scoring=scoring_class) #SVC does one-versus-one silently
print(pd.DataFrame({"C":[1],"Accuracy":[cv_iris["test_Accuracy"].mean()],
                    "Kappa":[cv_iris["test_Kappa"].mean()],
                    "AccuracySD":[cv_iris["test_Accuracy"].std(ddof=1)],
                    "KappaSD":[cv_iris["test_Kappa"].std(ddof=1)]}))
n_ir=len(yir)
lambdas_ir=np.logspace(-3,-1,5) #glmnet's multinomial lambda_max is a different
#formula, so just a log-spaced grid here
m_iris2=GridSearchCV(Pipeline([("scale",StandardScaler()),
                               ("m",LogisticRegression(solver="saga",max_iter=20000))]),
                     param_grid={"m__l1_ratio":np.linspace(0.1,1,5),"m__C":1/(n_ir*lambdas_ir)},
                     cv=foldsir,scoring=scoring_class,refit="Accuracy") #three classes, so
#LogisticRegression switches to multinomial, as glmnet does
m_iris2.fit(Xir,yir)
print(results_table(m_iris2).Accuracy.max())
print(m_iris2.best_estimator_["m"].classes_)

sys.stdout.file.close()
sys.stdout=sys.stdout.stdout
