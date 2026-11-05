import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


code_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(code_folder)
results_folder = os.path.join(project_folder, "results")


df = pd.read_csv(os.path.join(results_folder, "clean_data.csv"))


y = df["diabetes"]
X = df.drop(columns=["diabetes"])        


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

# Fit ONE model: logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy on the hidden 20%:
accuracy_percent = round(100 * model.score(X_test, y_test), 1)
n_patients = len(df)
percent_positive = round(100 * y.mean(), 1)   

# Figure: one bar per symptom showing how strongly the model uses it
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
plt.barh(coefficients.index, coefficients.values)
plt.xlabel("Model coefficient")
plt.title("Logistic regression coefficients")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "coefficients.png"), dpi=200)
plt.close()

# Save numbers in Python's own format (.pkl)
results = {"accuracy_percent": accuracy_percent,
           "n_patients": n_patients,
           "percent_positive": percent_positive}
with open(os.path.join(results_folder, "results.pkl"), "wb") as f:
    pickle.dump(results, f)

# Save the SAME numbers as LaTeX macros (.tex)
with open(os.path.join(results_folder, "results_macros.tex"), "w") as f:
    f.write("\\newcommand{\\testaccuracy}{" + str(accuracy_percent) + "}\n")
    f.write("\\newcommand{\\npatients}{" + str(n_patients) + "}\n")
    f.write("\\newcommand{\\percentpositive}{" + str(percent_positive) + "}\n")
print("Step 2 done: accuracy is", accuracy_percent)