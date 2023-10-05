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
