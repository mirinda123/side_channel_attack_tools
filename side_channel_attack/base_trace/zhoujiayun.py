#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sca 
@File    ：zhoujiayun.py
@Author  ：suyang
@Date    ：2022/5/21 12:57 
'''
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm, trange

# np.nanmax:排除nan值求最大
np.seterr(divide='ignore', invalid='ignore')
from numba import njit, jit

def getHW(byte):
    '''

    :param byte: input intermediate value
    :return: hw
    '''
    HW_table = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
                4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
                4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
                3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
                4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
                5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
                3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
                3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
                4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
                6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
                5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
                7, 7, 8]
    return HW_table[byte]
# all the attack need data matrix
def big_correlation_func_xiyj(n, url_trace, url_data):
    '''

    :param n: 变量的块数
    :param url_trace: 曲线集的路径
    :param url_data: 数据的路径
    :return:
    '''
    arr = np.load(url_trace + r"arrPart0.npy")
    # data = np.load(url_data + r"new_arrdata0.npy")
    Na = arr.shape[1]
    old_cov = np.zeros(Na)
    old_mean_data = 0
    old_mean_traces = np.zeros(Na)
    old_var_data = 0
    old_var_traces = np.zeros(Na)
    temp = 0
    for j in trange(n):
        data = np.load(url_data + r"xjyj{0}.npy".format(j))
        arr = np.load(url_trace + r"arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (temp + 1)
            new_mean_traces = old_mean_traces + (arr[i] - old_mean_traces) / (temp + 1)

            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                    temp + 1)
            new_var_traces = old_var_traces + (
                    (arr[i] - old_mean_traces) * (arr[i] - new_mean_traces) - old_var_traces) / (temp + 1)
            new_cov = (old_cov * temp + (data[i] - old_mean_data) * (arr[i] - new_mean_traces)) / (temp + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
            temp = temp + 1
    correlation_result = old_cov / np.sqrt(old_var_traces * old_var_data)
    return correlation_result
def big_correlation_func_sh(n, url_trace, url_data):
    '''

    :param n: 变量的块数
    :param url_trace: 曲线集的路径
    :param url_data: 数据的路径
    :return:
    '''
    arr = np.load(url_trace + r"arrPart0.npy")
    # data = np.load(url_data + r"new_arrdata0.npy")
    Na = arr.shape[1]
    old_cov = np.zeros(Na)
    old_mean_data = 0
    old_mean_traces = np.zeros(Na)
    old_var_data = 0
    old_var_traces = np.zeros(Na)
    temp = 0
    for j in trange(n):
        data = np.load(url_data + r"sh{0}.npy".format(j))
        arr = np.load(url_trace + r"arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (temp + 1)
            new_mean_traces = old_mean_traces + (arr[i] - old_mean_traces) / (temp + 1)
            # print(new_mean_traces)
            # print(traces[i])
            # print(traces[i] - new_mean_traces)
            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                    temp + 1)
            new_var_traces = old_var_traces + (
                    (arr[i] - old_mean_traces) * (arr[i] - new_mean_traces) - old_var_traces) / (temp + 1)
            # print(data[i][key_num])
            new_cov = (old_cov * temp + (data[i] - old_mean_data) * (arr[i] - new_mean_traces)) / (temp + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
            temp = temp + 1
    correlation_result = old_cov / np.sqrt(old_var_traces * old_var_data)
    return correlation_result
def function1(n):
    url_trace = r"H:/zhoujiayun1/"
    url_data = r"H:/zhoujiayun1/"

    # list = (np.loadtxt(url_data + r"aaaxiyj0.txt", delimiter=',', dtype="int")).tolist()
    for i in trange(n):
        databbb = np.loadtxt(url_data + r"aaash{0}.txt".format(i), delimiter=',', dtype="int")
        dataaaa = np.loadtxt(url_data + r"aaaxjyj{0}.txt".format(i), delimiter=',', dtype="int")
        for j in range(len(dataaaa)):
            dataaaa[j] = getHW(dataaaa[j])
            databbb[j] = getHW(databbb[j])
        np.save(url_data + r"xjyj{0}.npy".format(i), dataaaa)
        np.save(url_data + r"sh{0}.npy".format(i), databbb)
        # print(dataaaa)
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('aaaxiyj_cpa_traces')
    resultxiyj = big_correlation_func_xiyj(n, url_trace, url_data)
    ax[0].plot(resultxiyj)
    # plt.plot(result)



    ax[1].set_title('aaash_cpa_traces')
    resultsh = big_correlation_func_sh(n, url_trace, url_data)
    ax[1].plot(resultsh)
    plt.show()


def function2(n):
    url_trace = r"H:/zhoujiayun4/"
    url_data = r"H:/zhoujiayun4/"

    # list = (np.loadtxt(url_data + r"aaaxiyj0.txt", delimiter=',', dtype="int")).tolist()
    for i in trange(n):
        databbb = np.loadtxt(url_data + r"aaash{0}.txt".format(i), delimiter=',', dtype="int")
        dataaaa = np.loadtxt(url_data + r"aaaxjyj{0}.txt".format(i), delimiter=',', dtype="int")
        for j in range(len(dataaaa)):
            dataaaa[j] = getHW(dataaaa[j])
            databbb[j] = getHW(databbb[j])
        np.save(url_data + r"xjyj{0}.npy".format(i), dataaaa)
        np.save(url_data + r"sh{0}.npy".format(i), databbb)
        # print(dataaaa)
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('aaaxjyj_cpa_traces')
    resultxiyj = big_correlation_func_xiyj(n, url_trace, url_data)
    print(np.argmax(resultxiyj))
    ax[0].plot(resultxiyj)
    # plt.plot(result)

    ax[1].set_title('aaash_cpa_traces')
    resultsh = big_correlation_func_sh(n, url_trace, url_data)
    ax[1].plot(resultsh)
    plt.show()

if __name__ == '__main__':
    n = 100
    # list = (np.loadtxt(url_data + r"aaaxiyj0.txt", delimiter=',', dtype="int")).tolist()
    function2(n)

