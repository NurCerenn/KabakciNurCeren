#Toy stand-in for an R analysis workflow, in support of the workflows/latex unit
#of Biol 420/701: Machine Learning in Biology, University of Kansas.
#
#This script stands in for an entire R workflow (data import, cleaning, model
#fitting, and so on), which in a real project would be spread across several
#scripts invoked from a main.R. Here, the whole "analysis" is fitting a line to
#ten points. What matters is not the analysis but the pattern of what gets saved
#to the results directory, and how:
#
#1) output1 and output2 stand in for numeric results produced by the workflow
#   which need to be reported in the paper. They are saved twice: once as Rds
#   files (R's native format, so any later R code can reload them exactly), and
#   once as latex macros in a single .tex file, so the paper can display them
#   with no manual copying of numbers. The paper does \input{../results/NumericResults.tex}
#   in its preamble and then writes, e.g., $\outputone$ wherever the number goes.
#   If the analysis changes, re-run the code and re-compile the latex; the
#   numbers in the paper update automatically. Note that latex macro names may
#   contain only letters, so "output1" becomes \outputone. Rounding is done here,
#   at the point of writing the macro, so the paper gets a print-ready number.
#
#2) fig1.pdf stands in for whatever figure(s) the workflow produces. The paper
#   includes it with \includegraphics, again by file name, so a re-run of the
#   code updates the figure in the paper on the next compile.
#
#For tables: the same idea extends to data frames. The xtable package
#(https://cran.r-project.org/package=xtable) converts a data frame to a latex
#tabular environment which you save to a .tex file in results and \input into the
#paper. In python, the analogous tool is pandas.DataFrame.to_latex(); in Matlab, 
#there is no built-in, but latexTable 
#(https://www.mathworks.com/matlabcentral/fileexchange/44274-latextable)
#on the File Exchange is widely used. No table is made in this toy example.
#
#The python (code.py) and Matlab (code.m) scripts in this directory do exactly
#the same thing, and write files of the same names to results, except that the
#native-format saves use each language's own format (.Rds, .pkl, .mat). The
#paper never reads those, so their names can differ; the paper only reads
#fig1.pdf and NumericResults.tex, whose names are identical across languages.
#
#This script is meant to be run from the code directory, so relative paths to
#the results directory work.

#**basic setup

rm(list=ls())
graphics.off()

res_loc <- "../results/"
if (!dir.exists(res_loc)) #results is not committed to git, so create it if needed
{
  dir.create(res_loc)
}

#**the "analysis": fit a line to ten points

x <- 1:10
y <- 2*x + sin(x) #if these are your data, linear regression is a bit stupid, but
#this is a stand in for an analysis that would actually make sense
fit <- lm(y ~ x)
output1 <- unname(coef(fit)[2]) #slope - stands in for a numeric result to report
output2 <- unname(coef(fit)[1]) #intercept - stands in for another numeric result

#**save the numeric results in native format

saveRDS(output1, file=paste0(res_loc, "output1.Rds"))
saveRDS(output2, file=paste0(res_loc, "output2.Rds"))

#**save the figure

pdf(file=paste0(res_loc, "fig1.pdf"), width=5, height=4)
plot(x, y, pch=16, xlab="x", ylab="y")
abline(fit)
dev.off()

#**save the numeric results as latex macros

con <- file(paste0(res_loc, "NumericResults.tex"), open="w") #Open a connection
#to a file for writing, wiping any previous contents. 
cat(sprintf("\\newcommand{\\outputone}{%.3f}\n", output1), file=con)
#The \\ are because \ is a special character so you have to escape it. Gives one \ in output.
#The %.3f rounds to three decimal places. 
#The \n is a newline character, so each macro is on its own line.
#This defines a new latex macro called \outputone and the result of that is the numeric value.
cat(sprintf("\\newcommand{\\outputtwo}{%.3f}\n", output2), file=con)
close(con) #Close the file.
