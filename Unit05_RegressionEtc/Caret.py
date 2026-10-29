# I will use Yeast data for classification model in this assingment and I'm fetching it from UC Irvine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


columns = ["Sequence_Name", "mcg", "gvh", "alm", "mit",
           "erl", "pox", "vac", "nuc", "Localization"]
yeast=pd.read_csv("Unit05_RegressionETC/yeast.data", header=None, names=columns, sep=r"\s+")

yeast=yeast.drop(columns="Sequence_Name")
yeast['Localization']=yeast['Localization'].astype('category')

print(yeast.shape)
print(yeast.head())
print(yeast.info())
print(type(yeast))
print(yeast.isna())
print(yeast.isna().sum())


from pycaret.classification import ClassificationExperiment, setup
from sklearn.model_selection import train_test_split, KFold

yeast_val, yeast_test=train_test_split(yeast, test_size=0.25, stratify=yeast["Localization"], random_state=102)

exp=ClassificationExperiment()
exp.setup(data=yeast_val, test_data=yeast_test, target='Localization', 
          fold_strategy=KFold(n_splits=10,shuffle=True,random_state=102), normalize=True, session_id=103)


print(yeast_test.shape)
print(exp.get_config("X_test").columns.tolist())

ml=exp.models(internal=True)
print(ml)
print(ml[['Name']])

# now lets do model comperation
best=exp.compare_models(include=['lr','knn','dt','rf','svm','mlp','ridge','gbc','dummy','lightgbm','et','lda','qda','nb','rbfsvm'])
print(best)

# I choose to work with dummy, qda(quasritic discriminant analysis) and nb(naive bayes).
# Not only because they performed the best with defoults but also I'm not familiar with them either. Let's see how it goes

for name in ['dummy','qda','nb']:
    model=exp.create_model(name)
    print(name, model.get_params()) #to get parameters for each model

tuned_dummy = exp.tune_model(exp.create_model("dummy"), n_iter=10)
tuned_qda = exp.tune_model(exp.create_model("qda"), n_iter=10)
tuned_nb = exp.tune_model(exp.create_model("nb"), n_iter=10)

print(tuned_dummy.get_params())
print(tuned_qda.get_params())
print(tuned_nb.get_params())


#the loo~ooooo~oooo~oooo~p (spooky loop, happy hallowen!)

model_ids = ["dummy", "qda", "nb", "dt", "rf"]
rows=[]
tuned_models={}

for mid in model_ids:
    base=exp.create_model(mid, verbose=False)
    base_table=exp.pull()

    tuned=exp.tune_model(base, n_iter=10, optimize='Accuracy',
                         choose_better=False, verbose=False)
    tuned_table=exp.pull()

    tuned_models[mid]=tuned
    rows.append({
        'model': mid,
        "default_acc": base_table.loc["Mean", "Accuracy"],
        "tuned_acc": tuned_table.loc["Mean", "Accuracy"],
        "tuned_sd": tuned_table.loc["Std", "Accuracy"],
    })

results=pd.DataFrame(rows).sort_values('tuned_acc', ascending=False)
print(results)


# so here, the table tells me that all these models have better accuracy after tuning, except for the dummy, their accuracy after tuning went up. 
#based on the results table I decide to work with RF among the models

best_id = results.iloc[0]["model"]

base = exp.create_model(best_id, verbose=False)
big = exp.tune_model(base, n_iter=100, optimize="Accuracy",
                     choose_better=False, verbose=False)
big_table = exp.pull()

print("small grid :", results.iloc[0]["tuned_acc"])
print("large grid :", big_table.loc["Mean", "Accuracy"])
print("SD         :", big_table.loc["Std", "Accuracy"])

print(tuned_models["rf"].get_params())
print(big.get_params())
print(tuned_models["rf"].get_params() == big.get_params())

# Here comes the final model
# RF trained on yeas_val only. CV acc 0.617, SD 0.0223

final=tuned_models['rf']

# let's open the pandora's box, the vault

preds=exp.predict_model(final)
print(exp.pull())


from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(preds["Localization"], preds["prediction_label"])
print("test accuracy:", acc)
print(confusion_matrix(preds["Localization"], preds["prediction_label"]))