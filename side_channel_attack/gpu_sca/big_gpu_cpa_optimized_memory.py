import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import cuda, float32, int32
from scipy import stats

@cuda.jit()
# def cpa_last_round(data, traces, temp, old_var_data, old_var_traces, old_mean_data, old_mean_traces, old_cov, cor):
def cpa_last_round(data, traces, temp, new_mean_data, new_var_data, new_mean_traces, new_var_traces, new_cov, cor):
    '''

    :param data: 256列中间值数据
    :param traces: 采样曲线
    :param temp: 计数 N
    :param new_mean_data:
    :param old_mean_data:
    :param new_var_data:
    :param old_var_data:
    :param new_mean_traces:
    :param old_mean_traces:
    :param new_var_traces:
    :param old_var_traces:
    :param new_cov:
    :param old_cov:
    :param cor:
    :return:
    '''
    idx, idy = cuda.grid(2)
    if (idy >= traces.shape[1]) or (idx >= data.shape[1]):
        return
    for i in range(traces.shape[0]):
        # if idx == 0 and idy == 0:
        #     print(temp[idx,idy])
        # if idx == 1 and idy == 1:
        #     print(temp[idx,idy])
        # temp[idx,idy] 应该没有问题
        # new_mean_data 是存储某一列的均值
        old_mean_data = new_mean_data[idx, idy]
        old_mean_traces = new_mean_traces[idx, idy]
        new_mean_data[idx, idy] = new_mean_data[idx, idy] + (data[i, idx] - new_mean_data[idx, idy]) / (
                temp[idx, idy] + 1)
        # if idx == 0 and idy == 0:
        #     print("zhuyizheli")
        #     print("i", i)
        #     print("old_mean_data[idx,idy]", old_mean_data[idx, idy])
        #     print("data[i, idx]", data[i, idx])
        #     print("old_mean_data[idx,idy]", old_mean_data[idx, idy])
        #     print("temp[idx,idy]", temp[idx, idy])

        # if idx == 0 and idy == 0:
        #     print("i", i)
        #     print("new_mean_data[idx,idy]", new_mean_data[idx, idy])
        new_mean_traces[idx, idy] = new_mean_traces[idx, idy] + (traces[i, idy] - new_mean_traces[idx, idy]) / (
                temp[idx, idy] + 1)
        if idx == 0 and idy == 0:
            print("i", i)

            # 缩进有问题？？？
            # if i == 1:
            #     print("!!!old_var_traces[idx,idy]", old_var_traces[idx, idy])
            #     print("!!!traces[i, idy]", traces[i, idy])
            #     print("!!!old_mean_traces[idx,idy]", old_mean_traces[idx, idy])
            #     print("!!!new_mean_traces[idx,idy]", new_mean_traces[idx, idy])
            #     print("!!!temp[idx, idy]",temp[idx, idy])

            print("new_mean_traces[idx,idy]", new_mean_traces[idx, idy])
        new_var_traces[idx, idy] = new_var_traces[idx, idy] + (
                (traces[i, idy] - old_mean_traces) * (traces[i, idy] - new_mean_traces[idx, idy]) -
                new_var_traces[idx, idy]) / (temp[idx, idy] + 1)

        if idx == 0 and idy == 0:
            print("i", i)
            print("new_var_traces[idx,idy]", new_var_traces[idx, idy])
        new_var_data[idx, idy] = new_var_data[idx, idy] + (
                (data[i, idx] - old_mean_data) * (data[i, idx] - new_mean_data[idx, idy]) - new_var_data[
            idx, idy]) / (
                                         temp[idx, idy] + 1)
        if idx == 0 and idy == 0:
            print("i", i)
            print("new_var_data[idx,idy]", new_var_data[idx, idy])
        new_cov[idx, idy] = (new_cov[idx, idy] * i + (data[i, idx] - old_mean_data) * (
                traces[i, idy] - new_mean_traces[idx, idy])) / (temp[idx, idy] + 1)
        if idx == 0 and idy == 0:
            print("i", i)
            print("new_cov[idx, idy]:", new_cov[idx, idy])
        # 循环准入
        # old_cov[idx, idy] = new_cov[idx, idy]
        # old_var_traces[idx, idy] = new_var_traces[idx, idy]
        # old_var_data[idx, idy] = new_var_data[idx, idy]
        # old_mean_traces[idx, idy] = new_mean_traces[idx, idy]
        # old_mean_data[idx, idy] = new_mean_data[idx, idy]
        temp[idx, idy] = temp[idx, idy] + 1

    # 最终存储在cor
    cor[idx, idy] = new_cov[idx, idy] / math.sqrt(new_var_traces[idx, idy] * new_var_data[idx, idy])
    if (idx == 0 and idy == 0):
        print("i", i)
        print("cor[0, 0]", cor[0, 0])


@cuda.jit()
def cpa(data, traces, temp, new_mean_data,  new_var_data, new_mean_traces,
        new_var_traces, new_cov):
    idx, idy = cuda.grid(2)
    if (idy >= traces.shape[1]) or (idx >= data.shape[1]):
        return
    for i in range(traces.shape[0]):

        old_mean_data = new_mean_data[idx, idy]
        old_mean_traces = new_mean_traces[idx, idy]
        new_mean_data[idx, idy] = new_mean_data[idx, idy] + (data[i, idx] - new_mean_data[idx, idy]) / (
                temp[idx, idy] + 1)

        new_mean_traces[idx, idy] = new_mean_traces[idx, idy] + (traces[i, idy] - new_mean_traces[idx, idy]) / (
                temp[idx, idy] + 1)

        new_var_traces[idx, idy] = new_var_traces[idx, idy] + (
                (traces[i, idy] - old_mean_traces) * (traces[i, idy] - new_mean_traces[idx, idy]) -
                new_var_traces[idx, idy]) / (temp[idx, idy] + 1)

        new_var_data[idx, idy] = new_var_data[idx, idy] + (
                (data[i, idx] - old_mean_data) * (data[i, idx] - new_mean_data[idx, idy]) - new_var_data[
            idx, idy]) / (temp[idx, idy] + 1)
        new_cov[idx, idy] = (new_cov[idx, idy] * i + (data[i, idx] - old_mean_data) * (
                traces[i, idy] - new_mean_traces[idx, idy])) / (temp[idx, idy] + 1)
        temp[idx, idy] = temp[idx, idy] + 1




def cpa_function(n, url_trace, url_data):
    '''
    :instrctions:even  and  odd 间或采集
    :param n: blocks of traces
    :param N: sample numbers
    :return: result
    '''
    arr = np.load(url_trace + r"passarrPart0.npy")
    data = np.load(url_data + r"new_arrdata0.npy")
    # data = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # arr = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")

    cov_temp = np.zeros((data.shape[1], arr.shape[1]), dtype=np.uint32)

    new_var_data = np.zeros((data.shape[1], arr.shape[1]), dtype=np.float32)

    new_mean_data = np.zeros((data.shape[1], arr.shape[1]), dtype=np.float32)

    new_var_traces = np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float32)

    new_mean_traces = np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float32)

    new_cov = np.zeros((data.shape[1], arr.shape[1]), dtype=np.float32)


    gpu_temp = cuda.to_device(cov_temp)

    gpu_new_mean_data = cuda.to_device(new_mean_data)
    gpu_new_var_data = cuda.to_device(new_var_data)

    gpu_new_var_traces = cuda.to_device(new_var_traces)
    gpu_new_mean_traces = cuda.to_device(new_mean_traces)

    gpu_new_cov = cuda.to_device(new_cov)
    if n > 1:
        for j in trange(n - 1):
            arr = np.load(url_trace + r"passarrPart{0}.npy".format(j))
            data = np.load(url_data + r"new_arrdata{0}.npy".format(j))
            gpu_arr = cuda.to_device(arr)
            gpu_data = cuda.to_device(data)
            TPB = 16
            threads_per_block = (TPB, TPB)
            blocks_per_grid_x = int(math.ceil(data.shape[1] / threads_per_block[0]))
            blocks_per_grid_y = int(math.ceil(arr.shape[1] / threads_per_block[1]))
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            cpa[blocks_per_grid, threads_per_block](gpu_data, gpu_arr, gpu_temp, gpu_new_mean_data,
                                                    gpu_new_var_data, gpu_new_mean_traces,
                                                    gpu_new_var_traces, gpu_new_cov,
                                                    )
            # data, traces, temp, new_mean_data, old_mean_data, new_var_data, old_var_data, new_mean_traces, old_mean_traces,
            # new_var_traces, old_var_traces, new_cov, old_cov
            cuda.synchronize()
        arr = np.load(url_trace + r"passarrPart{0}.npy".format(n - 1))
        data = np.load(url_trace + r"new_arrdata{0}.npy".format(n - 1))
        # start in GPU
        gpu_arr = cuda.to_device(arr)
        gpu_data = cuda.to_device(data)
        C_global_mem = cuda.device_array((data.shape[1], arr.shape[1]))

        TPB = 16
        threads_per_block = (TPB, TPB)
        blocks_per_grid_x = int(math.ceil(data.shape[1] / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(arr.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        cpa_last_round[blocks_per_grid, threads_per_block](gpu_data, gpu_arr, gpu_temp, gpu_new_mean_data,
                                                           gpu_new_var_data, gpu_new_mean_traces,
                                                           gpu_new_var_traces, gpu_new_cov,
                                                           C_global_mem)

        cuda.synchronize()
        C_global_gpu = C_global_mem.copy_to_host()

        return C_global_gpu
    else:
        arr = np.load(url_trace + r"passarrPart{0}.npy".format(n - 1))

        data = np.load(url_data + r"new_arrdata{0}.npy".format(n - 1))

        gpu_arr = cuda.to_device(arr)
        gpu_data = cuda.to_device(data)
        C_global_mem = cuda.device_array((data.shape[1], arr.shape[1]))

        TPB = 16
        # C = cuda.const.array_like((arr.shape[1], data.shape[1]))
        threads_per_block = (TPB, TPB)
        blocks_per_grid_x = int(math.ceil(data.shape[1] / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(arr.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        cpa_last_round[blocks_per_grid, threads_per_block](gpu_data, gpu_arr, gpu_temp, gpu_new_mean_data,
                                                           gpu_new_var_data, gpu_new_mean_traces,
                                                           gpu_new_var_traces, gpu_new_cov,
                                                           C_global_mem)

        cuda.synchronize()

        C_global_gpu = C_global_mem.copy_to_host()
        return C_global_gpu


if __name__ == '__main__':
    import time

    n = 1
    url_trace = r"H:/order2random354besttrace12/"
    url_data = r"H:/order2random354besttrace12/"
    # A = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # B = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")
    # cpa_function(n, url_trace, url_data)
    # url_trace = url_data = r"G:/side_channel_attack/side_channel_attack/attack_method/traces/"
    start = time.time()
    A = cpa_function(n, url_trace, url_data)
    print(time.time() - start)
    print(A)
    # print("pearson result:", stats.pearsonr(url_data[:, 0], url_arr[:, 0]))
    # print("pearson result:", stats.pearsonr(url_data[:, 0], url_arr[:, 1]))
    # print("pearson result:", stats.pearsonr(url_data[:, 0], url_arr[:, 2]))
    plt.plot(A.T)
    plt.show()
