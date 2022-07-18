import numpy as np
import time
from tqdm import tqdm, trange
# np.nanmax:排除nan值求最大
np.seterr(divide='ignore', invalid='ignore')
from numba import njit, jit
# all the attack need data matrix


def correlation_func(data, traces):
    if data.ndim == 2 and traces.ndim == 2:
        old_cov = np.zeros(traces.shape[1])
        old_mean_data = 0
        old_mean_traces = np.zeros(traces.shape[1])
        old_var_data = 0
        old_var_traces = np.zeros(traces.shape[1])
        correlation_list = []
        for key_num in trange(data.shape[1]):
        # for key_num in trange(2):
            for i in range(traces.shape[0]):
                new_mean_data = old_mean_data + (data[i][key_num] - old_mean_data) / (i + 1)
                new_mean_traces = old_mean_traces + (traces[i] - old_mean_traces) / (i + 1)
                # print(new_mean_traces)
                # print(traces[i])
                # print(traces[i] - new_mean_traces)
                new_var_data = old_var_data + ((data[i][key_num] - old_mean_data) * (data[i][key_num] - new_mean_data) - old_var_data) / (i + 1)
                new_var_traces = old_var_traces + ((traces[i] - old_mean_traces) * (traces[i] - new_mean_traces) - old_var_traces) / (i + 1)
                new_cov = (old_cov * i + (data[i][key_num] - old_mean_data) * (traces[i] - new_mean_traces)) / (i + 1)
                old_mean_data = new_mean_data
                old_mean_traces = new_mean_traces
                old_cov= new_cov
                old_var_traces = new_var_traces
                old_var_data= new_var_data
            correlation_result = old_cov / np.sqrt(old_var_traces * old_var_data)
            # print(correlation_result)
            correlation_list.append(correlation_result)
        return np.array(correlation_list)

    else:
        print('please select proper style')
def big_correlation_func(n, url_trace,url_data):
    '''

    :param n: 变量的块数
    :param url_trace: 曲线集的路径
    :param url_data: 数据的路径
    :return:
    '''
    arr = np.load(url_trace + r"passarrPart0.npy")
    data = np.load(url_data + r"new_arrdata0.npy")
    Na = arr.shape[1]
    Nb = data.shape[1]
    old_cov = np.zeros(Na)
    old_mean_data = 0
    old_mean_traces = np.zeros(Na)
    old_var_data = 0
    old_var_traces = np.zeros(Na)
    correlation_list = []
    temp = 0
    # for key_num in trange(data.shape[1]):
    # for key_num in trange(2):
    for j in trange(n):
        data = np.load(url_data + r"new_arrdata{0}.npy".format(j))
        arr = np.load(url_trace + r"passarrPart{0}.npy".format(j))
        for key_num in range(data.shape[1]):
            for i in range(arr.shape[0]):
                new_mean_data = old_mean_data + (data[i][key_num] - old_mean_data) / (temp + 1)
                new_mean_traces = old_mean_traces + (arr[i] - old_mean_traces) / (temp + 1)
                # print(new_mean_traces)
                # print(traces[i])
                # print(traces[i] - new_mean_traces)
                new_var_data = old_var_data + ((data[i][key_num] - old_mean_data) * (data[i][key_num] - new_mean_data) - old_var_data) / (
                            temp + 1)
                new_var_traces = old_var_traces + (
                            (arr[i] - old_mean_traces) * (arr[i] - new_mean_traces) - old_var_traces) / (temp + 1)
                # print(data[i][key_num])
                new_cov = (old_cov * temp + (data[i][key_num] - old_mean_data) * (arr[i] - new_mean_traces)) / (temp + 1)
                old_mean_data = new_mean_data
                old_mean_traces = new_mean_traces
                old_cov = new_cov
                old_var_traces = new_var_traces
                old_var_data = new_var_data
                temp = temp+1

        correlation_result =  old_cov / np.sqrt(old_var_traces * old_var_data)
        # # print(correlation_result)
        correlation_list.append(correlation_result)
    return correlation_list

def find_key(k_num):
    for i in range(k_num):
        pass
    # one_key_guess_matrix = []
    # for i in range(data.shape[1]):
    #     n = correlation_func(data[:, i], traces)
    #     one_key_guess_matrix.append(n)
    # m = np.max(one_key_guess_matrix)
    # t = np.where(one_key_guess_matrix == m)
    # return t[0]


# np.argmax   np.where  np.max
if __name__ == '__main__':
    from numpy import loadtxt
    import math
    import matplotlib.pyplot as plt

    # data = np.load(r"G:/py+idea/python/sidechannel/attackmeasure/traces/SendData.npy")
    # trace = np.load(r"G:/py+idea/python/sidechannel/pretrace/mtraces/m0x01evenodd.npy")
    n = 200
    url_trace = r"H:/order2random354besttrace12/"
    url_data = r"H:/order2random354besttrace12/"
    # A = big_correlation_func(n, url_data, url_trace)
    # B = np.load(url_trace + r"passarrPart0.npy")
    # A = np.load(url_data + r"new_arrdata0.npy")
    # A = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # B = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")
    # print(data,trace)
    s_time = time.time()
    # cor = np.zeros((A.shape[1],B.shape[1]))
    # for i in trange(A.shape[1]):
    #     a = correlation_func(A[:,i], B)
    #     cor[i] = a
    # print(cor)
    a = big_correlation_func(n, url_trace,url_data)

    # A = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # B = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")

    # ddd = correlation_func(A,B)
    # np.save('G:/cpa_result.npy', ddd)
    e_time = time.time()
    print(e_time - s_time)

    plt.show()
