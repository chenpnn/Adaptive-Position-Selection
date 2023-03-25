import numpy as np
import talib
from talib.abstract import MACD, SMA

def normalize(x):
    '''
    truncate outlier w.r.t. median absolute deviation
    normalize x with Z-score method
    '''
    # x = np.array(x)
    if np.std(x) == 0:
        raise Exception('The input is a constant list.')
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    idx_upper = x > median + 3 * 1.4826 * mad
    idx_lower = x < median - 3 * 1.4826 * mad
    x[idx_upper] = median + 3 * 1.4826 * mad
    x[idx_lower] = median - 3 * 1.4826 * mad

    x = (x - np.mean(x)) / np.std(x)
    return x

def roc1(x):  # rate of change
    x = np.array(x)
    return talib.ROC(x, 1)

def macd(x):
    x = np.array(x)
    macd_, _, _ = talib.MACD(x)
    return macd_

def sma(x):  # Simple-Moving-Average of ROC1
    x = np.array(x)
    return talib.SMA(roc1(x), timeperiod=21)

def mom(x):
    x = np.array(x)
    return talib.MOM(x, timeperiod=21)

def std(x):
    x = np.array(x)
    return talib.STDDEV(x, timeperiod=21)

def slope(x):
    x = np.array(x)
    return talib.LINEARREG_SLOPE(x, timeperiod = 21)


func_list = [roc1, macd, sma, mom, std, slope]

if __name__ == '__main__':
    x = np.random.rand(100)
    for func in func_list:
        func(x)
