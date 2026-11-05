import os                # os helps to work with folders and file paths
import pandas as pd 

code_folder=os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(code_folder)              
data_folder = os.path.join(project_folder, "data")
results_folder = os.path.join(project_folder, "results")

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

df=pd.read_csv(os.path.join(data_folder,'diabetes_data_upload.csv'))

df.columns=df.columns.str.lower().str.replace(' ','_')

word_to_numbers={"Yes":1, 'No':0, 'Male':1, 'Female':0,
                 'Positive':1, 'Negative':0}

for column in df.columns:
    if column !='age':
        df[column] = df[column].map(word_to_numbers)


df.to_csv(os.path.join(results_folder, "clean_data.csv"), index=False)
print("Step 1 done:", len(df), "patients saved")