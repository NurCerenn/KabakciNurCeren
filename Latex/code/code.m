% *** AI-PRODUCED FILE (Claude, 2026-10-21), NOT REVIEWED OR RUN BY A HUMAN. USE WITH CARE. ***
% Note from Dan: I asked Claude to make this, as a resources, but I have not used Matlab for
% decades and don't have time to check it or even run it (I'd have to install Matlab). 
%
%Toy stand-in for a Matlab analysis workflow, in support of the workflows/latex
%unit of Biol 420/701: Machine Learning in Biology, University of Kansas.
%
%This is a Matlab version of code.R; see that file for the full explanation.
%
%This script stands in for an entire Matlab workflow (data import, cleaning,
%model fitting, and so on), which in a real project would be spread across
%several scripts or functions invoked from a main.m. Here, the whole "analysis"
%is fitting a line to ten points. What matters is not the analysis but the
%pattern of what gets saved to the results directory, and how:
%
%1) output1 and output2 stand in for numeric results produced by the workflow
%   which need to be reported in the paper. They are saved twice: once as .mat
%   files (Matlab's native format, so any later Matlab code can reload them
%   exactly), and once as latex macros in a single .tex file, so the paper can
%   display them with no manual copying of numbers. The paper does
%   \input{../results/NumericResults.tex} in its preamble and then writes, e.g.,
%   $\outputone$ wherever the number goes. If the analysis changes, re-run the
%   code and re-compile the latex; the numbers in the paper update
%   automatically. Note that latex macro names may contain only letters, so
%   "output1" becomes \outputone. Rounding is done here, at the point of writing
%   the macro, so the paper gets a print-ready number.
%
%2) fig1.pdf stands in for whatever figure(s) the workflow produces. The paper
%   includes it with \includegraphics, again by file name, so a re-run of the
%   code updates the figure in the paper on the next compile.
%
%For tables: Matlab has no built-in for writing a table as latex, but latexTable
%(https://www.mathworks.com/matlabcentral/fileexchange/44274-latextable) on the
%File Exchange is widely used; it writes a latex tabular environment which you
%save to a .tex file in results and \input into the paper. (In R the analogous
%tool is the xtable package; in python, pandas.DataFrame.to_latex().) No table
%is made in this toy example.
%
%The R (code.R) and python (code.py) scripts in this directory do exactly the
%same thing, and write files of the same names to results, except that the
%native-format saves use each language's own format (.Rds, .pkl, .mat). The
%paper never reads those, so their names can differ; the paper only reads
%fig1.pdf and NumericResults.tex, whose names are identical across languages.
%
%This script is meant to be run from the code directory, so relative paths to
%the results directory work.

%% **basic setup

clear all;
close all;

res_loc = "../results/";
if ~exist(res_loc, "dir") %results is not committed to git, so create it if needed
    mkdir(res_loc);
end

%% **the "analysis": fit a line to ten points

x = (1:10)';
y = 2*x + sin(x); %if these are your data, linear regression is a bit stupid, but
%this is a stand in for an analysis that would actually make sense
p = polyfit(x, y, 1); %p(1) is the slope, p(2) the intercept
output1 = p(1); %stands in for a numeric result to report
output2 = p(2); %stands in for another numeric result

%% **save the numeric results in native format

save(res_loc + "output1.mat", "output1");
save(res_loc + "output2.mat", "output2");

%% **save the figure

fig = figure("Units", "inches", "Position", [1 1 5 4]);
plot(x, y, "k.", "MarkerSize", 15);
hold on;
plot(x, output2 + output1*x, "k-");
xlabel("x");
ylabel("y");
exportgraphics(fig, res_loc + "fig1.pdf", "ContentType", "vector");
close(fig);

%% **save the numeric results as latex macros

fid = fopen(res_loc + "NumericResults.tex", "w");
fprintf(fid, "\\newcommand{\\outputone}{%.3f}\n", output1);
fprintf(fid, "\\newcommand{\\outputtwo}{%.3f}\n", output2);
fclose(fid);
