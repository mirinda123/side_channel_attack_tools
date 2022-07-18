from scipy import signal
import numpy as np
import time
from functools import lru_cache
# from enum import Enum as _Enum
# 此项为优化，暂时未用到
# from functools import lru_cache as _lru_cache
'''主要用到的滤波为低通高通带通，后期可能会扩展，使用开源scipy.signal'''
from numba import njit, jit
'生成低通滤波'
# a = signal.lfilter()

def LowPass(trace, frequency, cutoff,  axis=-1, precision='float32'):
    if not isinstance(trace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(frequency, int) and not isinstance(frequency, float):
        raise TypeError("'frequency' should be an of int or float type.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    b, a = signal.butter(3, 2 * cutoff / frequency, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, trace)
    return filtedData

def HighPass(trace, frequency, cutoff,  axis=-1, precision='float32'):
    if not isinstance(trace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(frequency, int) and not isinstance(frequency, float):
        raise TypeError("'frequency' should be an of int or float type.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    for value in cutoff:
        if not isinstance(value, int) and not isinstance(value, float):
            raise TypeError("'cutoff' should be a value or a collection of values of int or float type.")
        if value <= 0:
            raise ValueError("'cutoff' value(s) should be positive.")
        if len(cutoff) != 1:
            raise ValueError("cutoff should have 1 value.")
        if cutoff >= frequency / 2:
            raise ValueError("'cutoff' should be lower than frequency/2.")

    b, a = signal.butter(3, 2 * cutoff / frequency, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, trace)
    return filtedData


def BandPass(trace, frequency, cutoffmin, cutoffmax, axis=-1, precision='float32'):
    if not isinstance(trace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(frequency, int) and not isinstance(frequency, float):
        raise TypeError("'frequency' should be an of int or float type.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    for value in cutoffmin:
        if not isinstance(value, int) and not isinstance(value, float):
            raise TypeError("'cutoff' should be a value or a collection of values of int or float type.")
        if value <= 0:
            raise ValueError("'cutoff' value(s) should be positive.")
        if len(cutoffmin) != 1:
            raise ValueError("cutoff should have 1 value.")
    for value in cutoffmax:
        if not isinstance(value, int) and not isinstance(value, float):
            raise TypeError("'cutoff' should be a value or a collection of values of int or float type.")
        if value <= 0:
            raise ValueError("'cutoff' value(s) should be positive.")
        if len(cutoffmax) != 1:
            raise ValueError("cutoff should have 1 value.")

    b, a = signal.butter(3, [2 * cutoffmin / frequency, 2*cutoffmax/frequency],'bandpass')
    filtedData = signal.filtfilt(b, a, trace)
    return filtedData
# @jit
def fast_lowpass(traces,a):
    '''

    :param traces:
    :param a: 倍数
    :return:
    '''
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'trace' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0],traces.shape[1]), dtype=float)
    # point_list = []
    for t in range(traces.shape[0]):
        for i in range(0,traces.shape[1]):
            if i == 0:
                new_traces[t][0] = traces[t][0]
            else:
                new_traces[t][i] = (a*traces[t][i-1]+traces[t][i])/(1+a)
    return new_traces
if __name__ == '__main__':
    trace = np.load('G:/py+idea/python/sidechannel/pretrace/mtraces/mevenodd.npy')
    s_time = time.time()

    print(fast_lowpass(trace, 1))
    e_time = time.time()
    print(e_time - s_time)









