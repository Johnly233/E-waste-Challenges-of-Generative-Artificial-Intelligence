# E-waste-Challenges-of-Generative-Artificial-Intelligence
This is the program to realize global server E-waste prediction algorithm presented in paper 'E-waste Challenges of Generative Artificial Intelligence' The functionality of this program is to calucalte (predict) the number of servers with given training/inference scale parameters in the configurated period. The result contains average value, standard deviation and cumulative standard deviation from the starting time frame.

1.System requirements
The program is written and tested in Python3.9, win64 version. Any Python 3 version are compatible with this program.
The following Python libraries should be installed properly as well:
 - numpy
 - xlrd
 - pandas
 - openpyxl

2.Installation guide
Please install Python environment (preferably 3.9 or later version) and the Python libraries listed in System Requirements. The installation process follows standard Python installation steps and takes only a few minutes to complete.

3.Demo
The program comprises a 'Server prediction.py' Python script and a 'Demo data.xls' excel file. Configure the excel file to settle the scenarios and excute Python script with Python interpretor, the result file will be generated in the current folder. Detailed steps:

-A. Open 'Demo data.xls', where you can find 2 sheets. Sheet 'Input configuration' allows to change the value of algorithm required parameters: 'Number of random iterations, Average lifespan of server, Training time for one GAI model, Number of GPU per server, Number of GAI models, Upgrade strategy'. Please see detailed explaination of these parameters in Supplementary Information. The demo values (Baseline Scenario) are already filled in. You can change these values to study different scenarios. Sheet 'Computation data' contains future (predicted) data to support server number calculation, including 'Number of parameters per GAI model, Number of training data/token, Compuational power for training, Compuational power for inference, Global GAI user number' in the first 5 columns. The rest columns list out data in other scenarios for the reference. You can copy those data into first 5 columns to change scenario setting.

-B. Once 'Demo data.xls' is well set, run 'Server prediction.py' with installed Python interpretor. Wait a few seconds and the result file (named 'Result.xlsx') will be generated in the same folder.

-C. Open 'Result.xlsx'. It contains 4 columns. Column A refers to time index (by default, at seasonal interval). Column B refers to the mean value of server number estimation at this time frame. Column C refers to the standard deviation at this time frame and Column D refers to strandard deviation for cummulative amount from index 0 to current time index.

4.Instruction for use

