# IR_SF_ML

Instructions for the code

The code by default looks for a .csv file named Train_dataset.csv in the folder where the script is located.
The train dataset is comprised of X-ray irradiance data obtained from 6/9/17 through 10/9/17. In the research 
paper the train dataset was 6/9/17- 9/10/17 and the test dataset was 10/9/17.

The test file included with the code contains Ix values from 3/11/2008 and is intended for code testing purposes.

In order to make the code work, download the code and the datasets and put them, for example, on the Desktop
Two additional parameters are needed to set to make the code work. 

First, in line 111 add the path to the test .csv file, this can be the test file that was included with this code, 
or your own personal .csv file that contains an header with just Ix written in it and X-ray irradiance data below it. 
The minimum number of data points needed is 5 because of the features utilized to train the model. 

The form of the path to the test .csv file should be:

r"C:\Users\...\Test_dataset.csv"

The second parameter that the user needs to set is the z value located in line 165. The default value is 74 km
but if other value is needed it can be set. 

Needed libraries for the code to work:
  pandas,
  os,
  csv,
  numpy,
  xgboost,
  glob.

Python version 3.8.6

More detailed instructions on how to install Python and run the code:

1) Download Python from https://www.python.org/downloads/ (preferably version 3.8.6, but most probably newer versions will work)
2) When installing enable the option of adding Python to PATH
3) When the installation is complete, open the Command Prompt 
4) In the command prompt install the follwoing libraries as:
      a) pip install pandas,
      b) pip install numpy,
      c) pip install xgboost,
      d) pip install scikit-learn.
5) When all the libraries are installed search for IDLE on the search option on Windows
6) Open IDLE and open the script which is downloaded and placed on the desktop
7) When the script is open, adjust the parameters that were mentioned (path to testing file and z value if some other z value other than 74 km is needed)
8) Run the script and if everything is done successfully, the output of the script should be:
     data_statistical_features.csv- file where the statistical features are calculated
     data_flag_NaN.csv- file where empty cells are marked as NaN
     data_drop_NaN.csv- file where all rows containing at least one NaN values are dropped
     output_XGB.csv- output file with predictions. The predictions can be found in columns named Predicted_Beta and Predicted_Reflection_height. Aside from the features and target variables time delay i.e., slugishness and Ne are also presented in the output .csv file

