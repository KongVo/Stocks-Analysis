import pandas as pd
import os
import time
import math
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn import preprocessing
import yfinance as yf

#Adjust the size of matplotlib
import matplotlib as mpl
import matplotlib.image as mpimg
mpl.rc('figure', figsize=(8,7))
mpl.__version__
#Adjust the style of matplotlib
style.use('ggplot')

#get the current working directory
cwd = os.getcwd()

def savePlot(filename): 
    plt.savefig(cwd + "/" + filename + '.png')
    plt.close()
def printSavedMessage(message):
    print(message + " picture saved to current working directory" + "\n")
def plotPrediction(forecast_set,filename):
    last_date = df.iloc[-1,0]
    #print(last_date)
    last_unix = datetime.strptime(last_date, "%Y-%m-%d")
    next_unix = last_unix + timedelta(days=1)
    #print(next_unix)

    for i in forecast_set:
        next_date = next_unix
        next_unix += timedelta(days=1)
        #print(next_unix)
        #print(i)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[float(i)]
        #print(dfreg.loc[next_date])
        #print(type(dfreg.loc[next_date]))
          
    dfreg['Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    savePlot(filename)
    printSavedMessage(filename)

stockTicker = input("Enter a stock ticker(example: NFLX): ")
print(stockTicker)
ticker = yf.Ticker(stockTicker)

# get stock info
tickerInformation = ticker.info

# get historical market data
hist = ticker.history(period="max")

hist.to_csv(cwd + "/" + stockTicker +'.csv')

#Load a csv
df = pd.read_csv(cwd + "/" + stockTicker+ '.csv', delimiter = ',')

#print stock info
print(list(tickerInformation.items())[:])
print(df.tail())

#Rolling Mean
close_px = df['Close']
mavg = close_px.rolling(window=100).mean()
print("Moving Average determines trend"+ "\n")
print(mavg)

#Plot Moving Average
close_px.plot(label=str(stockTicker))
mavg.plot(label='mavg')
plt.legend()
savePlot("MovingAverage")
printSavedMessage("Moving Average Plot")

#Return Deviation
print("Return Deviation determines risk and return")
rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')
savePlot("ReturnDeviation")
printSavedMessage("Return Deviation Plot")

#Predicting Stocks Price
print("Feature Engineering for High Low % and % Change")
dfreg = df.loc[:,['Close','Volume']]
dfreg['HighLow_%'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['%Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
print(dfreg)

#Pre-processing & Cross Validation
#Drop missing value
dfreg.fillna(value=-99999, inplace=True)
#separate 1 percent of data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
#separate label, predict the AdjClos
forecast_col = 'Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))
#scale the X for linear regression
X = preprocessing.scale(X)
#find Data Series of late X & early X for model generation & evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
#separate label & identify as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

x_train = np.array(X).reshape(len(X), -4000)
y_train = np.array(y).reshape(len(y), -4000)
x_test = np.array(X).reshape(len(X), -400)
y_test = np.array(y).reshape(len(y), -400)

print("Simple Linear Analysis shows a linear relationship between two or more variables")
#Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(x_train, y_train)
#Evaluate Linear regression
confidencereg = clfreg.score(x_test, y_test)
#Result Linear regression
print('The linear regression confidence: ' + str(confidencereg))
#stock forecast Linear regression
Linear_Regression_forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
#Plot prediction(Linear_Regression_forecast_set)
plotPrediction(Linear_Regression_forecast_set,"Linear_Regression_Prediction")


print("Quadratic Discriminant Analysis (QDA) has polynomials and produces curves")
#Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
#Evaluate
clfpoly2.fit(x_train, y_train)
#Result
confidencepoly2 = clfpoly2.score(x_test, y_test)
print('The quadratic regression 2 confidence: ' + str(confidencepoly2))
#stock forecast Quadratic Regression 2
Quadratic_Regression_2_forecast_set = clfpoly2.predict(X_lately)
dfreg['Forecast'] = np.nan
#Plot prediction(Quadratic_Regression_2_forecast_set)
plotPrediction(Quadratic_Regression_2_forecast_set,"Quadratic_Regression_2_Prediction")

#Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
#Evaluate
clfpoly3.fit(x_train, y_train)
#Result
confidencepoly3 = clfpoly3.score(x_test, y_test)
print('The quadratic regression 3 confidence: ' + str(confidencepoly3))
#stock forecast Quadratic Regression 3
Quadratic_Regression_3_forecast_set = clfpoly3.predict(X_lately)
dfreg['Forecast'] = np.nan
#Plot prediction(Quadratic_Regression_3_forecast_set)
plotPrediction(Quadratic_Regression_3_forecast_set,"Quadratic_Regression_3_Prediction")

print("K Nearest Neighbor (KNN) uses feature similarity to predict values of data points")
#KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
#Evaluate
clfknn.fit(x_train, y_train)
#Result
confidenceknn = clfknn.score(x_test, y_test)
print('The knn regression confidence: ' + str(confidenceknn))
#stock forecast KNN Regression
KNN_Regression_forecast_set = clfknn.predict(X_lately)
dfreg['Forecast'] = np.nan
#Plot prediction(KNN Regression_forecast_set)
plotPrediction(KNN_Regression_forecast_set,"KNN_Regression_Prediction")


k=input("press close to exit") 
