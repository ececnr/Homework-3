from math import sqrt
import pandas as pd
import numpy as np, numpy.random
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt  # ,ExponentialSmoothing,
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import random

# The following lines are to suppress warning messages.
import warnings

warnings.filterwarnings("ignore")


# Functions needed
def decomp(frame, name, f, mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array, freq=f, model=mod, two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show()  # Uncomment to reshow plot, saved as Figure 1.

    return result

def decomp0605(name, f, mod='Additive'):
    array = name
    result = sm.tsa.seasonal_decompose(array, freq=f, model=mod, two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show()  # Uncomment to reshow plot, saved as Figure 1.
    return result

def test_train(series, i1, i2, i3):
    sarray = series
    trainarray= sarray[i1:i2]
    testarray = sarray[i2:i3]
    return testarray

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=60).mean()
    rolstd = pd.Series(timeseries).rolling(window=60).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.savefig("_Test_Stationary_RollingMean&StandartDeviation.png")
    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array, copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def RMSE(value, estimate):
    ms = mean_squared_error(value, estimate, sample_weight=None, multioutput="uniform_average")
    rmse = sqrt(ms)
    return rmse

def Error_output(estimate1, estimate2, value1, value2):
    estimate1557=[]
    estimate1558=[]
    estimate1557.append(estimate1)
    estimate1558.append(estimate2)

    Error_1557 = RMSE(value1, estimate1557)
    Error_1558 = RMSE(value2, estimate1558)
    
    return Error_1557, Error_1558

def estimate_holt(array, alpha, slope , forecast):
    numbers = array
    model = Holt(numbers)
    fit = model.fit(alpha, slope)
    estimate = fit.forecast(forecast)[-1]
    return estimate

def RMSE(y_true,y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def estimate_ses(testarray, alpha):
    numbers = testarray
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha, optimized=False).forecast(1)
    return estimate

def get_small(freq, a, b):
    series = tradedata[seriesname]
    shortdata = []
    for i in range (0, freq):
        k = a+(i*60)
        shortdata.append(series[k])
        j = b+(i*60)
        shortdata.append(series[j]) 
        i += 1
    return shortdata

def moving_average(df,seriesname,windowsize):
    movingavg = df[seriesname].rolling(windowsize).mean()
    return movingavg

#check these rows
def sum_weights(sarray):
    weightedlist = []
    x=0
    while x <len(sarray):
        if x < len(sarray)-59:
            r = [random.random() for i in range(0,59)]
            s = sum(r)
            r = [ i/s for i in r ]
            r.sort()
            result = numpy.dot( sarray[x:x+59], r)
        else:
            r = [random.random() for i in range(0,len(sarray)-x)]
            s = sum(r)
            r = [ i/s for i in r ]
            r.sort()
            result = numpy.dot( sarray[x:len(sarray)], r)
        weightedlist.append(result)
        x +=1
    return weightedlist

def Sum_error(error1, error2):
    totalerror = error1 + error2
    return totalerror
    


#end of checking rows

# 1)a-
# stationarity testing for complete data

def a_1(tradedata , index):

    series = tradedata[seriesname]
    test_stationarity(series)
    result = decomp(tradedata, seriesname, f=1)
    test_stationarity(result.trend)

    # result: all of the data is stationary according to plot


    series = tradedata[seriesname]
    sarray = np.asarray(series)

    size = len(series)
    train = series[0:index]   # 8640 , 5760
    trainarray = np.asarray(train)
    test = series[index:]     # 8640 , 5760
    testarray = np.asarray(test, dtype=float)
    print("Training data:", trainarray, "Test data:", testarray)

    # stationarity testing for 6th of may

    test_stationarity(testarray )
    result = decomp0605(testarray, f=360)
    test_stationarity(result.trend)

# 1)-b
# estimation for 15:57 and 15:58 for 6th of may
# method 1: SES estimation
def b_1(tradedata):
    series = tradedata[seriesname]
    sarray = np.asarray(series)
    # last three days of data is used for simplicity with deducting the last two data to forecast the 15:57
    size = len(series)
    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-2]
    testarray1 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray1)

    series = tradedata[seriesname]
    sarray = np.asarray(series)
    # last three days of data is used for simpliccity with deducting the last data to forecast the 15:58
    size = len(series)
    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-1]
    testarray2 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray2)


    # Function for Simple Exponential Smoothing

    alpha = 0.025
    forecast = 1
    ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4 )
    print("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)

    ses_06051558 = round(estimate_ses(testarray2, alpha)[0], 4 )
    print("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)

    # method 2: # Trend estimation with Holt


    alpha = 0.038
    slope = 0.1
    forecast = 1
    # alpha and slope adjusted to find the closest values possible


    holtfor1557 = round(estimate_holt(testarray1, alpha, slope , forecast), 4)
    print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)

    holtfor1558 = round(estimate_holt(testarray2, alpha, slope , forecast), 4)
    print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)

    ##chechking for RMSE


    print(tradedata["Time"] == "3:57:00 PM")
    RMSE_for_holtfor = RMSE([series[11037],series[11038]],[holtfor1557,holtfor1558])

    RMSE_for_ses= RMSE([series[11037],series[11038]],[ses_06051557 , ses_06051558])


    if (RMSE_for_holtfor > RMSE_for_ses):
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.025
        forecast = 1
        ses_06051559 = round(estimate_ses(testarray1, alpha ,forecast )[0], 4 )
        print("Simple Exponential Smoothing estimation for 6.May 15:59", ses_06051559)

        forecast = 2
        ses_06051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 6.May 16:00", ses_06051600)

        forecast = 1440
        ses_07051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 7.May 16:00", ses_07051600)

    else :
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.038
        slope = 0.1

        forecast = 1
        holtfor1559 = round(estimate_holt(testarray1, alpha, slope , forecast), 4)
        print("Holt trend estimation with alpha for 6.May 15:59 =", alpha, ", and slope =", slope, ": ", holtfor1559)

        forecast = 2
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 6.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)

        forecast = 1440
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 7.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)

# 1-c

def c_1(tradedata ) :

    new_trade_data = tradedata[tradedata['Volume'] != 0]

    a_1(new_trade_data, 5760)
    b_1(new_trade_data)

    print(new_trade_data)

def a_2(data):
    new_trade_data = data
    series = new_trade_data[seriesname]
    test_stationarity(series,)
    result = decomp(new_trade_data, seriesname, f=1)
    test_stationarity(result.trend)

def b_2(data):
    freq=183
    endhourseries = get_small(freq, 57, 58)
    
    series = endhourseries

    test = series[:-2]
    testarray1 = np.asarray(test)

    test = series[:-1]
    testarray2 = np.asarray(test)

    alpha = 0.025
    forecast = 1
    ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)

    ses_06051558 = round(estimate_ses(testarray2, alpha)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)

    alpha = 0.038
    slope = 0.1
    forecast = 1
    # alpha and slope adjusted to find the closest values possible

    holtfor1557 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)

    holtfor1558 = round(estimate_holt(testarray2, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)

    print(tradedata)

    RMSE_for_holtfor = RMSE([tradedata['Close'][len(tradedata)-2], tradedata['Close'][len(tradedata)-1]], [holtfor1557, holtfor1558])

    RMSE_for_ses = RMSE([tradedata['Close'][len(tradedata)-2], tradedata['Close'][len(tradedata)-1]] , [ses_06051557, ses_06051558])

    print("RMSE HF -> ", RMSE_for_holtfor ,"RMSE SES -> ", RMSE_for_ses)

def c_2(data):
    tradedata = pd.read_csv("trade.txt", sep = "\t")
    tradedata["period"] = tradedata["Day"].map(str) + tradedata["Time"]
    tradedata.set_index("period")
    tradedata["Volume"].replace(['0', '0.0'], '', inplace=True) # discharge zero
    tradedata = tradedata.fillna(method = "ffill")
    tradedata = tradedata.fillna(method = "bfill")
    seriesname = "Close"
    freq=183
    data = get_small(freq, 57, 58)
    series = np.asarray(data)
    image_name="C_2"
    test_stationarity(series)
    result = decomp0605(series, f=1)
    test_stationarity(result.trend)
    b_2(data)



#Starting check these rows

def a_3(tradedata) :
    new_trade_data = tradedata[tradedata['Volume'] != 0]
    image_name="3_a"
    series = moving_average(tradedata, seriesname, 60)
    test_stationarity(series)
    result = decomp0605(series, f=1)
    test_stationarity(result.trend)

    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-2]
    testarray1 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray1)

    series = tradedata[seriesname]
    sarray = np.asarray(series)
    # last three days of data is used for simpliccity with deducting the last data to forecast the 15:58
    size = len(series)
    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-1]
    testarray2 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray2)

    # Function for Simple Exponential Smoothing

    alpha = 0.025
    forecast = 1
    ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)

    ses_06051558 = round(estimate_ses(testarray2, alpha)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)

    # method 2: # Trend estimation with Holt

    alpha = 0.038
    slope = 0.1
    forecast = 1
    # alpha and slope adjusted to find the closest values possible

    holtfor1557 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)

    holtfor1558 = round(estimate_holt(testarray2, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)

    print(tradedata)

    RMSE_for_holtfor = RMSE([tradedata['Close'][len(tradedata) - 2], tradedata['Close'][len(tradedata) - 1]],
                            [holtfor1557, holtfor1558])

    RMSE_for_ses = RMSE([tradedata['Close'][len(tradedata) - 2], tradedata['Close'][len(tradedata) - 1]],
                        [ses_06051557, ses_06051558])

    if (RMSE_for_holtfor > RMSE_for_ses):
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.025
        forecast = 1
        ses_06051559 = round(estimate_ses(testarray1, alpha ,forecast )[0], 4 )
        print("Simple Exponential Smoothing estimation for 6.May 15:59", ses_06051559)

        forecast = 2
        ses_06051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 6.May 16:00", ses_06051600)

        forecast = 1440
        ses_07051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 7.May 16:00", ses_07051600)

    else :
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.038
        slope = 0.1

        forecast = 1
        holtfor1559 = round(estimate_holt(testarray1, alpha, slope , forecast), 4)
        print("Holt trend estimation with alpha for 6.May 15:59 =", alpha, ", and slope =", slope, ": ", holtfor1559)

        forecast = 2
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 6.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)

        forecast = 1440
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 7.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)


def b_3(tradedata) :
    series = tradedata[seriesname]
    array = np.asarray(series)
    image_name="b_3"

    weightedlist = sum_weights(array)
    sarray = weightedlist
    test_stationarity(sarray)
    result = decomp0605(sarray,f=360)
    test_stationarity(result.trend)
    print("result:data is not stationary according to dickey-fuller test")

    #estimation for 15:57 and 15:58 for 6th of may
    #method 1: SES estimation
    print("last five days of data is used for simplicity to forecast")
    testarray1 = test_train(sarray,0, 4320, -2)
    testarray2 = test_train(sarray,0, 4320, -1)

    # Function for Simple Exponential Smoothing
    alpha = 0.025   
    ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4)
    print ("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)
    print("\n")
    ses_06051558 = round (estimate_ses(testarray2, alpha)[0], 4)
    print ("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)
    print("\n")

    #method 2: # Trend estimation with Holt
    alpha = 0.038
    slope = 0.1
    forecast = 1
    #alpha and slope adjusted to find the closest values possible
    holtfor1557 = round(estimate_holt(testarray1,alpha, slope, forecast),4)
    print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)
    print("\n")
    holtfor1558 = round(estimate_holt(testarray2,alpha, slope, forecast),4)
    print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)
    print("\n")

    #chechking for RMSE

    value1557=[]
    value1558=[]
    value15_57= sarray[-2]
    value15_58= sarray[-1]
    value1557.append(value15_57)
    value1558.append(value15_58)



    #error for SES
    Error_ses1557, Error_ses1558 = Error_output(ses_06051557, ses_06051558, value1557, value1558)
    print("RMSE for SES estimation for 15:57 and 15:58 in order are;", Error_ses1557,",", Error_ses1558)
    print("\n")
    #error for holt
    Error_holt1557, Error_holt1558 = Error_output(holtfor1557, holtfor1558, value1557, value1558)
    print("RMSE for holt estimation for 15:57 and 15:58 in order are;", Error_holt1557,",", Error_holt1558)
    print("\n")

    total_error_for_ses = Sum_error(Error_ses1557,Error_ses1558)
    total_error_for_holt = Sum_error(Error_holt1557,Error_holt1558)

    print("error in ses and error in holt in order are", total_error_for_ses, total_error_for_holt)
    print("\n")

    if total_error_for_ses < total_error_for_holt:
        print ("use ses for future forecast")
    else:
        print("use holt for future forecast")
     
    print("\n")
    # estimating 15:59
    testarray3= test_train(sarray,0, 4320, len(series))
    alpha = 0.038
    slope = 0.1
    forecast= 1
    holtfor1559 = round(estimate_holt(testarray3,alpha, slope, forecast),4)
    print("Holt trend estimation for 15:59 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1559)
    print("\n")
    forecast= 2
    holtfor1600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
    print("Holt trend estimation for 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1600)
    print("\n")
    forecast= 1440
    holtfor071600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
    print("Holt trend estimation for 07.05.2019 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor071600)
    print("\n")

def c_3(tradedata):
    series = tradedata[seriesname]
    image_name="c_3"
    choose_random(tradedata)
    test_stationarity(series)
    result = decomp0605(sarray,f=360)
    test_stationarity(result.trend)

    

    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-2]
    testarray1 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray)

    series = tradedata[seriesname]
    sarray = np.asarray(series)
    # last three days of data is used for simpliccity with deducting the last data to forecast the 15:58
    size = len(series)
    train = series[0:5760]
    trainarray = np.asarray(train)
    test = series[5760:-1]
    testarray2 = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray)

    # Function for Simple Exponential Smoothing

    alpha = 0.025
    forecast = 1
    ses_06051557 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)

    ses_06051558 = round(estimate_ses(testarray2, alpha, forecast)[0], 4)
    print("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)

    # method 2: # Trend estimation with Holt

    alpha = 0.038
    slope = 0.1
    forecast = 1
    # alpha and slope adjusted to find the closest values possible

    holtfor1557 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)

    holtfor1558 = round(estimate_holt(testarray2, alpha, slope, forecast), 4)
    print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)

    ##chechking for RMSE

    print(tradedata["Time"] == "3:57:00 PM")
    RMSE_for_holtfor = RMSE([series[11037], series[11038]], [holtfor1557, holtfor1558])

    RMSE_for_ses = RMSE([series[11037], series[11038]], [ses_06051557, ses_06051558])

    if (RMSE_for_holtfor > RMSE_for_ses):
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.025
        forecast = 1
        ses_06051559 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 6.May 15:59", ses_06051559)

        forecast = 2
        ses_06051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 6.May 16:00", ses_06051600)

        forecast = 1440
        ses_07051600 = round(estimate_ses(testarray1, alpha, forecast)[0], 4)
        print("Simple Exponential Smoothing estimation for 7.May 16:00", ses_07051600)

    else:
        test = series[5760:]
        testarray1 = np.asarray(test)
        alpha = 0.038
        slope = 0.1

        forecast = 1
        holtfor1559 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 6.May 15:59 =", alpha, ", and slope =", slope, ": ", holtfor1559)

        forecast = 2
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 6.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)

        forecast = 1440
        holtfor1600 = round(estimate_holt(testarray1, alpha, slope, forecast), 4)
        print("Holt trend estimation with alpha for 7.May 16:00 =", alpha, ", and slope =", slope, ": ", holtfor1600)
#End of checking rows

tradedata = pd.read_csv("trade.txt", sep="\t")
seriesname = "Close"
tradedata["period"] = tradedata["Day"].map(str) + tradedata["Time"]
tradedata.set_index("period")
tradedata["Volume"].replace(['0', '0.0'], '', inplace=True) # discharge zero
tradedata = tradedata.fillna(method = "ffill")
tradedata = tradedata.fillna(method = "bfill")

a_1(tradedata, 8740)
b_1(tradedata)
c_1(tradedata)
a_2(tradedata)
b_2(tradedata)
c_2(tradedata)
a_3(tradedata)
b_3(tradedata)


