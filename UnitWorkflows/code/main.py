# main.py
# Author: NUR CEREN KABAKCI
# Purpose: runs the whole analysis. Use:  python main.py 
# Reads:  data/diabetes_data_upload.csv
# Writes: clean data, a figure, a .pkl and a .tex file into "results"


import runpy

runpy.run_path("UnitWorkflows/code/read_and_clean_data.py")   # step 1: import and clean
runpy.run_path("UnitWorkflows/code/analysis.py")       # step 2: model, figure, numbers

print("All done! Check the results folder.")