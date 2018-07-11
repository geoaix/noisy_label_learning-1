createDataFile.py file produce random point data in setting range. They
are clean_100.txt, clean_1000.txt, clean_10000.txt, and clean_100000.txt.
We can add testing or taining to the clean data file name, so they can
be use for learning optimized parameters.

There are three main files in this folder. They are main_risk.py, 
main_clean_risk_mesh.py, and main_plot_clean_risk_w_in_diff_rou.py.

main_risk.py find the optimized parameter W for the decision function 
f hat. And use the optimized function to train the training data and 
test the testing data. The risks results of training data and testing 
data in different rou are plotted in a diagram which is shown in final
report Figure 2.

main_clean_risk_mesh.py computes all optimized parameter W and those
Ws are used to compute the clean data risk in different rou settings.
And then, the result is save to clean_risk.txt. Be careful for the 
input training file name is different to the testing file name. For
example, in my setting: clean_file_name = 'clean_1000_training.txt';
and clean_testing_data = readTextFile('clean_1000_testing.txt'; two
files must be different. 

main_plot_clean_risk_w_in_diff_rou.py plots the file saved by 
main_clean_risk_mesh.py and the data of the file is plotted in a 3D 
diagram (Figure three in the final report) which shows that the most 
area are flat, and this is consistent to the result of lemma one. The
file name which save by main_clean_risk_mesh.py must be the same with
the file name to be plot in main_plot_clean_risk_w_in_diff_rou.py.

