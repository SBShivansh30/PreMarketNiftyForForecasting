# %% [markdown]
# Importing libraries

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# Class and it's corresponding methods for training and testing

# %%
class RegressionModels:
    def __init__(self):
        self.ridge_model = Ridge()
        self.svr_model = SVR()
        self.rf_model = RandomForestRegressor()
        self.selected_features = None

    def train(self, X_train, y_train):
        self._feature_selection(X_train, y_train)
        self.ridge_model.fit(X_train[self.selected_features], y_train)
        self.svr_model.fit(X_train[self.selected_features], y_train)
        self.rf_model.fit(X_train[self.selected_features], y_train)

    def _feature_selection(self, X_train, y_train):
        model = Ridge()
        selector = RFE(estimator=model, n_features_to_select=3)
        selector.fit(X_train, y_train)
        self.selected_features = X_train.columns[selector.support_]

    def predict(self, X_test):
        ridge_pred = self.ridge_model.predict(X_test[self.selected_features].values.reshape(1, -1))
        svr_pred = self.svr_model.predict(X_test[self.selected_features].values.reshape(1, -1))
        rf_pred = self.rf_model.predict(X_test[self.selected_features].values.reshape(1, -1))

        return ridge_pred, svr_pred, rf_pred

    def evaluate(self, y_pred, y_test_actual):
        ridge_pred, svr_pred, rf_pred= y_pred

        ridge_error = np.abs(ridge_pred - y_test_actual)
        svr_error = np.abs(svr_pred - y_test_actual)
        rf_error = np.abs(rf_pred - y_test_actual)

        print("Ridge Regression Error:", ridge_error)
        print("SVR Error:", svr_error)
        print("Random Forest Error:", rf_error)
        
        ## to identify which one gives least error
        best_model = min(ridge_error, svr_error, rf_error)
        if best_model == ridge_error:
            return "Ridge Regression"
        elif best_model == svr_error:
            return "SVR"
        elif best_model == rf_error:
            return "Random Forest"


# %% [markdown]
# Fetching Data

# %%
# Fetching data
data = pd.read_excel('Training&TestingData.xlsx')
print(data.columns.values)
data


# %%
###building null dataframe to for storing forecasted values
forecast_df = pd.DataFrame(index=['Ridge Regression', 'SVR', 'Random Forest'], 
                        columns=set(data.columns)-set(['NIFTY Pre-Open', 'ADV/decline ratio', 'SGX NIFTY % gain at 9:08 IST','Date']))


# %%
forecast_df

# %%
##Running a loop to train the model for different times of the day
for i in set(data.columns)-set(['NIFTY Pre-Open', 'ADV/decline ratio', 'SGX NIFTY % gain at 9:08 IST','Date']):
    df = pd.DataFrame(data[['NIFTY Pre-Open', 'ADV/decline ratio', 'SGX NIFTY % gain at 9:08 IST', i]],
                    columns=['NIFTY Pre-Open', 'ADV/decline ratio','SGX NIFTY % gain at 9:08 IST', i])


    # Split the dataframe into training and test data for CROSS - VALIDATION
    train_data = df.iloc[:2]
    test_data = df.iloc[2]

    X_train = train_data.iloc[:, :3]
    y_train = train_data.iloc[:, 3].values.reshape(-1, 1)  # Reshape target variable to 2D array
    X_test = test_data.iloc[:3]
    y_test_actual = test_data.iloc[3]

    
    models = RegressionModels()

    #Train the models
    models.train(X_train, y_train)

    # 
    #Calling prediction method in the class
    y_pred = models.predict(X_test)

    # forecasted values df
    # print('x train columns', set(df.columns)-set(X_train.columns))
    # forecast_df = pd.DataFrame(index=['Ridge Regression', 'SVR', 'Random Forest', 'Correlation Model'], 
    #                         columns=set(data.columns)-set(X_train.columns)-set(['Date']))
    # print(y_pred)
    # Add the forecasted values to the dataframe
    forecast_df.loc['Ridge Regression',i] = y_pred[0][0][0]
    forecast_df.loc['SVR',i] = y_pred[1][0]
    forecast_df.loc['Random Forest',i] = y_pred[2][0]


    # print('Actual Value')
    # print(y_test_actual)

    # print("Forecasted Values:")

forecast_df[['NIFTY % gain at 10 AM', 'NIFTY % gain at 12 PM',
       'NIFTY % gain at 2 PM', 'NIFTY % gain at 3:30 PM']]


# %% [markdown]
# ACTUAL VALUES FOR 2nd June

# %%
pd.DataFrame(data.loc[2,:]).T[['NIFTY % gain at 10 AM', 'NIFTY % gain at 12 PM',
       'NIFTY % gain at 2 PM', 'NIFTY % gain at 3:30 PM']]

# %% [markdown]
# FORECASTED RESULTS

# %%
forecast_df[['NIFTY % gain at 10 AM', 'NIFTY % gain at 12 PM',
       'NIFTY % gain at 2 PM', 'NIFTY % gain at 3:30 PM']]

# %% [markdown]
# Building a Box Plot

# %%
import plotly.express as px
import pandas as pd

# Example time series data for our three days on 15 min candle
mkt_data = pd.read_excel('marketdata.xlsx')

day1_data = mkt_data[20230531]  # Use column name instead of index
day2_data = mkt_data[20230601]
day3_data = mkt_data[20230602]

custom_points = {'Day': ['31st May', '1st June', '2nd June'],
                 'Pre-market NIFTY Level': [18560, 18517, 18551]}

# Combine the data into a single dataframe
data = pd.DataFrame({'31st May': day1_data, '1st June': day2_data, '2nd June': day3_data})
custom_data = pd.DataFrame(custom_points)

# Convert the dataframe into "long" format
data_long = pd.melt(data, var_name='Day', value_name='NIFTY Level')

# Generate the advanced box plot using plotly
fig = px.box(data_long, x='Day', y='NIFTY Level', points='all')

# Add custom points using scatter trace
# fig.add_trace(px.scatter(custom_data, x='Day', y='Value',color_discrete_sequence=['red'], size_max=10000 ).data[0])
scatter =px.scatter(custom_data, x='Day', y='Pre-market NIFTY Level',color_discrete_sequence=['red'], size_max=10000 ).data[0]
scatter.name = 'Pre-market Value'
fig.add_trace(scatter)

# fig.update_traces(color='red')
fig['data'][0]['showlegend']=True
fig['data'][0]['name'] = 'Market Data 15M'
fig['data'][1]['showlegend']=True
fig['data'][1]['name'] = 'Pre-market Value'

# print(fig['data'])
# Set the plot title and axes labels
fig.update_layout(title='Box Plot of NIFTY Data and Pre-market level',
                  xaxis_title='Day',
                  yaxis_title='NIFTY Level')

# Display the plot
fig.show()



