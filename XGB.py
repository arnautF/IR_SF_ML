#Information:

#Python version v.3.8.6

#Workflow for research paper titled "Ionospheric Response on Solar Flares through Machine Learning Modelling" submitted for publishing in Universe.
#Code written by: Filip Arnaut
#Affiliation: Institute of Physics Belgrade, University of Belgrade, Pregrevica 118, 11080 Belgrade, Republic of Serbia
#Contact email: filip.arnaut@ipb.ac.rs

#Introduction:
#This code accompanies the research paper "Ionospheric Response on Solar Flares through Machine Learning Modelling" and is intended to serve as the code from which similar results
#as those in the research paper can be obtained

#In the folder where this script can be found are train and test .csv files that can be utilized with this code

#Comments are given in the code below to which function does what
#read the arguments needed for each function before using

#basic overview:
    #input the path for the pre-prcessed train dataset
    #input the path for the non pre-processed test dataset
    #the X-ray irradiance data in the test .csv file should have a header with just "Ix" in it
    #the script will pre-process the test dataset and train the model on the train dataset and make predictions on the test dataset
    #the script makes 4 .csv outputs:
                                      #data_statistical_features.csv- file where the statistical features are calculated
                                      #data_flag_NaN.csv- file where empty cells are marked as NaN
                                      #data_drop_NaN.csv- file where all rows containing at least one NaN values are dropped
                                      #output_XGB.csv- output file with predictions. The predictions can be found in columns named Predicted_Beta and Predicted_Reflection_height
                                      #aside from the features and target variables time delay i.e., slugishness and Ne are also presented in the output .csv file

##############################################################################################################################################################################################################################################################################

import pandas as pd
import os
import csv
import numpy as np
import xgboost as xgb
import glob

# Script 1: Calculate statistical features
def calculate_parameters(csv_file, output_file):
    df = pd.read_csv(csv_file)

    # Convert columns to numeric data type
    df['Ix'] = pd.to_numeric(df['Ix'], errors='coerce')
    df['lagged_1_Ix'] = df['Ix'].shift(1)
    df['lagged_2_Ix'] = df['Ix'].shift(2)
    df['lagged_3_Ix'] = df['Ix'].shift(3)
    df['lagged_4_Ix'] = df['Ix'].shift(4)
    df['lagged_5_Ix'] = df['Ix'].shift(5)
    df['roll_mean_5_Ix'] = df['Ix'].rolling(5).mean()
    df['roll_median_5_Ix'] = df['Ix'].rolling(5).median()
    df['roll_stdev_5_Ix'] = df['Ix'].rolling(5).std()
    df['first_diff_Ix'] = df['Ix'].diff()
    df['second_diff_Ix'] = df['first_diff_Ix'].diff()
    df['rate_of_change_Ix'] = df['Ix'].pct_change()
    
    df.to_csv(output_file, index=False)

# Script 2: Flag Empty Cells
def flag_empty_cells(input_file, output_file):
    with open(input_file, 'r') as file:
        data = list(csv.reader(file))
        header = data[0]
        data = data[1:]

        for i, row in enumerate(data, start=1):
            for j in range(len(row)):
                if not row[j].strip():
                    row[j] = 'NaN'

            row.insert(0, i)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index'] + header)
        writer.writerows(data)

# Script 3: Process CSV and Flag NaN values
def process_csv_and_flag_nans(input_file, output_file):
    df = pd.read_csv(input_file)
    df.replace('', np.nan, inplace=True)
    df['NaN_check'] = df.isnull().any(axis=1).astype(int)
    df.to_csv(output_file, index=False, na_rep='NaN')

# Script 4: Drop Rows with NaN Flag
def drop_rows_with_nan_flag(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df[df['NaN_check'] != 1]
    df.to_csv(output_file, index=False)

# Function to train XGBoost model and make predictions
def train_and_predict_xgboost(X_train, y_train, X_test):
    # Log transform the target variables
    y_train_log = np.log1p(y_train)

    # Initialize and train the regression model on log-transformed targets
    model = xgb.XGBRegressor(n_estimators=150, eta=0.2) #n_estimators and eta (learning rate) can be changed
    model.fit(X_train, y_train_log)

    # Predict on the test set
    y_pred_log = model.predict(X_test)

    # De-transform the predictions
    y_pred = np.expm1(y_pred_log)

    return y_pred

# Main function to call the scripts and train/predict with XGBoost
def main():
    input_file = #file path to train dataset

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file_statistical = os.path.join(script_directory, "data_statistical_features.csv")
    output_file_flag_nans = os.path.join(script_directory, "data_flag_NaN.csv")
    output_file_drop_nans = os.path.join(script_directory, "data_drop_NaN.csv")

    # Script 1: Calculate Parameters
    calculate_parameters(input_file, output_file_statistical)

    # Script 2: Flag Empty Cells
    flag_empty_cells(output_file_statistical, output_file_flag_nans)

    # Script 3: Process CSV and Flag NaN values
    process_csv_and_flag_nans(output_file_flag_nans, output_file_drop_nans)

    # Script 4: Drop Rows with NaN Flag
    drop_rows_with_nan_flag(output_file_drop_nans, output_file_drop_nans)

    # Load train and test data
    train_file_path = os.path.join(script_directory, "Train_dataset.csv")
    train_data = pd.read_csv(train_file_path)

    test_file_path = os.path.join(script_directory, "data_drop_NaN.csv")
    test_data = pd.read_csv(test_file_path)

    # Features and targets
    features = ['Ix',
                'lagged_1_Ix',
                'lagged_2_Ix',
                'lagged_3_Ix',
                'lagged_4_Ix',
                'lagged_5_Ix',
                'roll_mean_5_Ix',
                'roll_median_5_Ix',
                'roll_stdev_5_Ix',
                'first_diff_Ix',
                'second_diff_Ix',
                'rate_of_change_Ix']
    targets = ['Beta', 'Reflection_height']

    # Split train and test data
    X_train, y_train = train_data[features], train_data[targets]
    X_test = test_data[features]

    # Train XGBoost model and predict
    y_pred = train_and_predict_xgboost(X_train, y_train, X_test)

    # Create a DataFrame for predictions including features
    predictions_df = pd.DataFrame(np.concatenate([X_test, y_pred.reshape(-1, len(targets))], axis=1),
                                columns=features + [f'Predicted_{target}' for target in targets])

    # Calculate time_delay and Ne based on equations
    predictions_df['time_delay [min]'] = 0.45385 - 0.44863 * np.log10(predictions_df['Ix'])
    z=74 #Adjust the z value, default is 74 km
    predictions_df['Ne [m-3]'] = 1.43 * 10**13 * np.exp(-0.15 * predictions_df['Predicted_Reflection_height']) * np.exp((predictions_df['Predicted_Beta'] - 0.15) * (z - predictions_df['Predicted_Reflection_height']))

    # Output data with predictions to a CSV file
    output_csv_path = os.path.join(script_directory, "output_XGB.csv")
    predictions_df.rename(columns={'Ix': 'Ix [W/m3]'}, inplace=True)
    predictions_df.rename(columns={'Predicted_Beta': 'Predicted_Beta [km-1]'}, inplace=True)
    predictions_df.rename(columns={'Predicted_Reflection_height': 'Predicted_Reflection_height [km]'}, inplace=True)
    predictions_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
