# NIFTY Forecasting

This repository contains code for forecasting NIFTY values using machine learning models. The aim is to understand how well the NIFTY pre-market snapshot sustains as the market progresses and build a model to independently predict NIFTY values at different times during the day.

## Introduction
The code provided in this repository trains and evaluates three regression models: Ridge Regression, Support Vector Regression (SVR), and Random Forest Regression. These models are used to predict the NIFTY percentage gain at different times during the day based on pre-market snapshot data of NIFTY and SGX NIFTY (approx 9:08 AM IST).

## Installation
To run the code, you need to have Python and the following libraries installed:
- pandas
- numpy
- scikit-learn

Clone this repository and install the required dependencies using the following command:

## Usage
Update the Training&TestingData.xlsx file with your own data.
Run the PreMarkettraining.py script or PreMarkettraining.ipynb (prefereably this only to see output better in jupyter notebook) to train and evaluate the regression models.
The results will be printed, showing the forecasted values for each model in predicting the NIFTY values.

## Results
The evaluated models are Ridge Regression, SVR, and Random Forest Regression. The results may vary based on the dataset and the number of training data points available. In this specific case, due to the limited training data (2 days), the model's performance may not be reliable. Further evaluation with a larger dataset is recommended.
