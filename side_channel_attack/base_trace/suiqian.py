#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sca 
@File    ：suiqian.py.py
@Author  ：suyang
@Date    ：2022/5/6 16:06 
'''

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from tqdm import tqdm, trange
from numba import cuda
import numba as nb

np.seterr(divide='ignore', invalid='ignore')


def t_test(n, url_trace, trace_name):
    '''
    :instrctions:even  and  odd 间或采集
    :param n: blocks of traces
    :param N: sample numbers
    :return: result
    '''
    # arr = np.load(url_trace + r"2process_arrPart0.npy")
    arr = np.load(url_trace + trace_name + r"0.npy")
    count = 0
    N = arr.shape[1]
    old_var_even = np.zeros(N)
    old_mean_even = np.zeros(N)
    old_var_odd = np.zeros(N)
    old_mean_odd = np.zeros(N)
    oddcount = 0
    evencount = 0
    for j in trange(n):
        # arr = np.load(url_trace + r"2process_arrPart{0}.npy".format(j))
        arr = np.load(url_trace + trace_name + r"{0}.npy".format(j))
        for i in range(arr.shape[0]):
            if count % 2 == 0:
                new_mean = old_mean_even + (arr[i] - old_mean_even) / (evencount + 1)
                new_var = old_var_even + ((arr[i] - old_mean_even) * (arr[i] - new_mean) - old_var_even) / (
                        evencount + 1)
                old_mean_even = new_mean
                old_var_even = new_var
                evencount += 1
                # print(evencount)
                count = count + 1
            else:
                new_mean = old_mean_odd + (arr[i] - old_mean_odd) / (oddcount + 1)
                new_var = old_var_odd + ((arr[i] - old_mean_odd) * (arr[i] - new_mean) - old_var_odd) / (oddcount + 1)
                old_mean_odd = new_mean
                old_var_odd = new_var
                oddcount += 1
                count = count + 1
    temp1 = old_mean_even - old_mean_odd
    temp2 = (old_var_even / evencount) + (old_var_odd / oddcount)
    test_result = temp1 / np.sqrt(temp2)
    return test_result

@nb.jit


def moving_resamples(traces, k):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1] // k), dtype=float)
    if k >= traces.shape[1]:
        raise ValueError('window is too bigooo!')
    n = traces.shape[1]
    for t in range(traces.shape[0]):
        list = []
        # //地板除
        for i in range(n // k):
            sum = 0
            for j in range(i * k, i * k + k):
                sum = traces[t][j] + sum
            a = sum / k
            list.append(a)
        new_traces[t] = np.array(list)
    return new_traces


@nb.jit
def to_one2(traces):
    if not isinstance(traces, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1]), dtype=float)
    a = np.mean(traces, axis=0)
    for t in range(traces.shape[0]):
        # s = np.std(traces[t])
        new_traces[t] = (traces[t] - a)
    return new_traces


# @cuda.jit
def LowPass(trace, frequency, cutoff, axis=-1, precision='float32'):
    if not isinstance(trace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(frequency, int) and not isinstance(frequency, float):
        raise TypeError("'frequency' should be an of int or float type.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    b, a = signal.butter(3, 2 * cutoff / frequency, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, trace)
    return filtedData


if __name__ == '__main__':

    # 文件路径
    url_trace = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/"

    # 文件块数
    num_file_blocks = 57

    # 预处理
    for j in trange(num_file_blocks):
        arr = np.load(url_trace + r"blinkyTestOneAndarrPart{0}.npy".format(j))
        tracetoone = moving_resamples(arr, 10)
        tracepass1 = to_one2(tracetoone)
        tracepass = LowPass(tracepass1, 25e6, 2e6)
        np.save(url_trace + r'process_arrPart{0}.npy'.format(j), tracepass)

    # 原始曲线
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('initial_traces')
    ax[0].axhline(y=4.5, ls='--', c='red', linewidth=2)
    ax[0].axhline(y=-4.5, ls='--', c='red', linewidth=2)
    result = t_test(num_file_blocks, url_trace, "blinkyTestOneAndarrPart")
    np.save(url_trace+r'aes2.npy',result)
    ax[0].plot(result)
    a = []
    for i in range(len(result)):
        if abs(result[i])>=4.5:
            a.append((i*8)/21)
    # print(a)
    # 预处理之后
    ax[1].axhline(y=4.5, ls='--', c='red', linewidth=2)
    ax[1].axhline(y=-4.5, ls='--', c='red', linewidth=2)
    ax[1].set_title('after_pre_process')
    result2 = t_test(num_file_blocks, url_trace, "process_arrPart")

    ax[1].plot(result2)
    plt.show()

