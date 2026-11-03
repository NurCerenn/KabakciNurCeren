#Toy stand-in for a python analysis workflow, in support of the workflows/latex
#unit of Biol 420/701: Machine Learning in Biology, University of Kansas.
#
#This is a python version of code.R; see that file for the full explanation.
#It was produced by an AI (Claude) at the instructor's request, and has been
#run and checked by comparing its output with that of code.R.
#
#This script stands in for an entire python workflow (data import, cleaning,
#model fitting, and so on), which in a real project would be spread across
#several modules invoked from a main.py. Here, the whole "analysis" is fitting a
#line to ten points. What matters is not the analysis but the pattern of what
#gets saved to the results directory, and how:
#
#1) output1 and output2 stand in for numeric results produced by the workflow
#   which need to be reported in the paper. They are saved twice: once as pickle
#   files (python's native format, so any later python code can reload them
#   exactly), and once as latex macros in a single .tex file, so the paper can
#   display them with no manual copying of numbers. The paper does
#   \input{../results/NumericResults.tex} in its preamble and then writes, e.g.,
#   $\outputone$ wherever the number goes. If the analysis changes, re-run the
#   code and re-compile the latex; the numbers in the paper update
#   automatically. Note that latex macro names may contain only letters, so
#   "output1" becomes \outputone. Rounding is done here, at the point of writing
#   the macro, so the paper gets a print-ready number.
#
#2) fig1.pdf stands in for whatever figure(s) the workflow produces. The paper
#   includes it with \includegraphics, again by file name, so a re-run of the
#   code updates the figure in the paper on the next compile.
#
#For tables: pandas.DataFrame.to_latex() converts a data frame to a latex
#tabular environment which you save to a .tex file in results and \input into
#the paper. (In R the analogous tool is the xtable package; in Matlab,
#latexTable on the File Exchange.) No table is made in this toy example.
#
#The R (code.R) and Matlab (code.m) scripts in this directory do exactly the
#same thing, and write files of the same names to results, except that the
#native-format saves use each language's own format (.Rds, .pkl, .mat). The
#paper never reads those, so their names can differ; the paper only reads
#fig1.pdf and NumericResults.tex, whose names are identical across languages.
#
#This script is meant to be run from the code directory, so relative paths to
#the results directory work.

# %% **basic setup

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("pdf") #no display needed
import matplotlib.pyplot as plt

res_loc = "../results/"
if not os.path.isdir(res_loc): #results is not committed to git, so create it if needed
    os.mkdir(res_loc)

# %% **the "analysis": fit a line to ten points

x = np.arange(1, 11)
y = 2*x + np.sin(x) #if these are your data, linear regression is a bit stupid, but
#this is a stand in for an analysis that would actually make sense
slope, intercept = np.polyfit(x, y, 1)
output1 = slope #stands in for a numeric result to report
output2 = intercept #stands in for another numeric result

# %% **save the numeric results in native format

with open(res_loc + "output1.pkl", "wb") as f:
    pickle.dump(output1, f)
with open(res_loc + "output2.pkl", "wb") as f:
    pickle.dump(output2, f)

# %% **save the figure

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(x, y, "o", color="black")
ax.plot(x, intercept + slope*x, "-", color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig(res_loc + "fig1.pdf")
plt.close(fig)

# %% **save the numeric results as latex macros

with open(res_loc + "NumericResults.tex", "w") as f:
    f.write(f"\\newcommand{{\\outputone}}{{{output1:.3f}}}\n")
    f.write(f"\\newcommand{{\\outputtwo}}{{{output2:.3f}}}\n")
