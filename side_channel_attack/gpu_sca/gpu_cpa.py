import cmath
import math

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import cuda, float32, int32
import time
import scared
import psutil


#
# @cuda.jit('void(float64[:,:], float32[:])')
# def vec_mean_col(vecs, mean):
#     idx = cuda.grid(1)
#     if idx >= N:
#         return
#     temp = 0
#     old_mean = 0
#     for i in range(NV):
#         new_mean = old_mean + (vecs[i, idx] - old_mean) / (temp + 1)
#         old_mean = new_mean
#         temp = temp + 1
#     mean[idx] = new_mean


# @cuda.jit('void(float64[:,:], float32[:])')
# def vec_var_col(vecs, var):
#     idx = cuda.grid(1)
#     if idx >= N:
#         return
#     temp = 0
#     old_var = 0
#     old_mean = 0
#     for i in range(NV):
#         new_mean = old_mean + (vecs[i, idx] - old_mean) / (temp + 1)
#         new_var = old_var + ((vecs[i, idx] - old_mean) * (vecs[i, idx] - new_mean) - old_var) / (temp + 1)
#         old_var = new_var
#         old_mean = new_mean
#         temp = temp + 1
#     var[idx] = new_var


@cuda.jit('void(float64[:],float64[:,:], float64[:])')
def vec_cor_col(data, traces, cor):
    idx = cuda.grid(1)
    if (idx >= 1250):
        return
    temp = 0
    old_var_data = 0
    old_var_traces = 0
    old_mean_data = 0
    old_mean_traces = 0
    old_cov = 0
    for i in range(25000):
        # mean
        new_mean_data = old_mean_data + (data[i] - old_mean_data) / (temp + 1)
        print(new_mean_data)
        new_mean_traces = old_mean_traces + (traces[i, idx] - old_mean_traces) / (temp + 1)
        # print(new_mean_traces)
        # var
        new_var_traces = old_var_traces + (
                    (traces[i, idx] - old_mean_traces) * (traces[i, idx] - new_mean_traces) - old_var_traces) / (
                                     temp + 1)
        new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                    temp + 1)

        new_cov = (old_cov * i + (data[i] - old_mean_data) * (traces[i, idx] - new_mean_traces)) / (temp + 1)
        # 循环准入
        old_cov = new_cov
        old_var_traces = new_var_traces
        old_var_data = new_var_data
        old_mean_traces = new_mean_traces
        old_mean_data = new_mean_data
        temp = temp + 1
    cor[idx] = old_cov / math.sqrt(old_var_traces * old_var_data)





# @cuda.jit('void(float64[:],float64[:,:], float32[:])')
# 这个是正确运行的
@cuda.jit()
def cpa(data, traces, cor):
    idx, idy = cuda.grid(2)
    if (idy >= traces.shape[1]) or (idx  >= data.shape[1]):
        return
    # if (idy < traces.shape[1]) and (idx < data.shape[1]):
    # temp = 0
    old_var_data = 0
    old_var_traces = 0
    old_mean_data = 0
    old_mean_traces = 0
    old_cov = 0
    for i in range(data.shape[0]):
        # mean
        new_mean_data = old_mean_data + (data[i, idx] - old_mean_data) / (i + 1)
        new_mean_traces = old_mean_traces + (traces[i, idy] - old_mean_traces) / (i + 1)

        new_var_traces = old_var_traces + (
                (traces[i, idy] - old_mean_traces) * (traces[i, idy] - new_mean_traces) - old_var_traces) / (
                                 i + 1)
        new_var_data = old_var_data + (
                    (data[i, idx] - old_mean_data) * (data[i, idx] - new_mean_data) - old_var_data) / (
                               i + 1)
        new_cov = (old_cov * i + (data[i, idx] - old_mean_data) * (traces[i, idy] - new_mean_traces)) / (i + 1)
        # 循环准入
        old_cov = new_cov
        old_var_traces = new_var_traces
        old_var_data = new_var_data
        old_mean_traces = new_mean_traces
        old_mean_data = new_mean_data
        # temp = temp + 1
    cor[idx, idy] = old_cov / math.sqrt(old_var_traces * old_var_data)


# peform row-gpu_sca
# @cupy.fuse()


def get_imediate(label, klength):
    k_array = np.arange(2 ** klength)
    data = label * k_array
    return data
def main():

    # perform column-gpu_sca
    start = time.time()
    A = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # A = np.load(r"H:/order2random354besttrace12/new_arrdata0.npy")
    B_ = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")
    # B_ = np.load(r"H:/order2random354besttrace12/passarrPart0.npy")
    B = np.ascontiguousarray(B_)

    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)

    C_global_mem = cuda.device_array((A.shape[1], B.shape[1]))
    TPB = 16

    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = int(math.ceil(A.shape[1] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    # start = time.time()
    cpa[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)
    # 等待所有内核计算完成
    cuda.synchronize()


    cuda.synchronize()

    C_global_gpu = C_global_mem.copy_to_host()


    end = time.time()
    # print(end - start)
    print(C_global_gpu)
    plt.plot(C_global_gpu.T)
    plt.show()


if __name__ == '__main__':
    main()
    print("0000000",psutil.Process().memory_info().peak_wset)
