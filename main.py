# Jessica Chen
# BMW Lab
# 10/08/2020

# Following along to a tutorial on using LSTM in TF 2.0 to predict
# power consumption: https://kgptalkie.com/multi-step-time-series-predicting-using-rnn-lstm/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import nan
import datetime as dt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

"""
# read dataset
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates=True, low_memory=False)

# print out first 5 rows
# print(data.head())

# concatenate data and time columns to 'date_time' column
data['date_time'] = data['Date'].str.cat(data['Time'], sep=' ')
data.drop(['Date', 'Time'], inplace=True, axis=1)

# set first index column to date_time column
data.set_index(['date_time'], inplace=True)

# replace all '?' missing values with a NaN float
data.replace('?', nan, inplace=True)

# makes data just one array of floating point values, as opposed to a bunch of mixed types
data = data.astype('float')


# return info on dataset
# print(data.info())

# check null values
# print(np.isnan(data).sum())


# fill missing values with values from 24 hours ago
def fill_missing(data):
    one_day = 24 * 60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row - one_day, col]


fill_missing(data.values)

# check nan values
# print(np.isnan(data).sum())
# print(data.info())
# print(data.shape)
# print(data.head())

# save cleaned-up version of dataset into new file

# convert dataframe to .csv
data.to_csv('cleaned_data.csv')
"""

# read dataset
dataset = pd.read_csv('cleaned_data.csv', parse_dates=True, index_col='date_time', low_memory=False)

# print top rows
# print(dataset.head())

# print bottom rows
# print(dataset.tail())

# downsample data into day-wise bins and sum values of timestamps in each bin
data = dataset.resample('D').sum()

# data after sampling into daywise manner
# print(data.head())

# plot all features

"""
fig, ax = plt.subplots(figsize=(18,18))

for i in range(len(data.columns)):
    plt.subplot(len(data.columns), 1, i+1)
    name = data.columns[i]
    plt.plot(data[name])
    plt.title(name, y=0, loc='right')
    plt.yticks([])
plt.show()
fig.tight_layout()
"""

# plot active power consumption for each year
years = ['2007', '2008', '2009', '2010']
"""
fig, ax = plt.subplots(figsize=(18, 18))

for i in range(len(years)):
    plt.subplot(len(years), 1, i + 1)
    year = years[i]
    active_power_data = data[str(year)]
    plt.plot(active_power_data['Global_active_power'])
    plt.title(str(year), y=0, loc='left')
plt.show()
fig.tight_layout()

# show table for year 2006
print(data['2006'])
"""

# plot year-wise global_active_power with histogram
"""
fig, ax = plt.subplots(figsize=(18, 18))

for i in range(len(years)):
    plt.subplot(len(years), 1, i + 1)
    year = years[i]
    active_power_data = data[str(year)]
    active_power_data['Global_active_power'].hist(bins=200)
    plt.title(str(year), y=0, loc='left')
plt.show()
fig.tight_layout()

"""

# histogram plot for all features
"""
fig, ax = plt.subplots(figsize=(18, 18))

for i in range(len(data.columns)):
    plt.subplot(len(data.columns), 1, i + 1)
    name = data.columns[i]
    data[name].hist(bins=200)
    plt.title(name, y=0, loc='right')
    plt.yticks([])
plt.show()
fig.tight_layout()
"""

# histogram plot for consumption in each month of 2007

"""
new_data = dataset['2007']
print(new_data.head())

new_data['date_time'] = pd.to_datetime(new_data['date_time'], format='%Y-%m-%d %H:%M:%S')

months = [i for i in range(1, 13)]

fig, ax = plt.subplots(figsize=(18, 18))

for i in range(len(months)):
    ax = plt.subplot(len(months), 1, i + 1)
    active_power_data = new_data[
        (new_data['date_time'].dt.month == months[i]) & (new_data['date_time'].dt.year == 2007)]
    active_power_data['Global_active_power'].hist(bins=100)
    ax.set_xlim(0, 5)
    plt.title('2007-' + str(months[i]), y=0, loc='right')
plt.show()
fig.tight_layout()
"""

# split data into training and testing datasets
data_train = data.loc[:'2009-12-31', :]['Global_active_power']
# print(data_train.head())

data_test = data['2010']['Global_active_power']
# print(data_train.head())

# check number of datapoints in each set
# print(data_train.shape)
# print(data_test.shape)

# convert data into numpy array
data_train = np.array(data_train)

# split data into weeks
X_train, y_train = [], []

for i in range(7, len(data_train) - 7):
    X_train.append(data_train[i - 7:i])
    y_train.append(data_train[i:i + 7])

# convert into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# return shape of arrays
# print(X_train.shape, y_train.shape)

# print y_train
# print(pd.DataFrame(y_train).head())

# normalize dataset between 0 and 1
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)

# print(pd.DataFrame(X_train).head())

# convert to 3D array
X_train = X_train.reshape(1098, 7, 1)

# build sequential model using Keras
reg = Sequential()
reg.add(LSTM(units=200, activation='relu', input_shape=(7, 1)))
reg.add(Dense(7))

# loss = mean square error, optimizer = adam
reg.compile(loss='mse', optimizer='adam')

# train model
reg.fit(X_train, y_train, epochs=100)

# make testing dataset in numpy array
data_test = np.array(data_test)

# split test data by weeks
X_test, y_test = [], []

for i in range(7, len(data_test) - 7):
    X_test.append(data_test[i - 7:i])
    y_test.append(data_test[i:i + 7])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = x_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

X_test = X_test.reshape(331, 7, 1)

y_pred = reg.predict(X_test)

# bring y_pred values to their original forms using inverse transform
y_pred = y_scaler.inverse_transform(y_pred)
# print(y_pred)

# inverse transform y_test to y_true
y_true = y_scaler.inverse_transform(y_test)
# print(y_true)


def evaluate_model(y_true, y_predicted):
    scores = []

    # calculate scores for each day
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)

    # calculate score for whole prediction
    total_score = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
    total_score = np.sqrt(total_score//(y_true.shape[0]*y_predicted.shape[1]))

    return total_score, scores


print(evaluate_model(y_true, y_pred))

# standard deviation
print(np.std(y_true[0]))