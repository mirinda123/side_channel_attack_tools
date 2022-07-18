# import numba.cuda
import numba
import numpy as np

# a = np.zeros((1000000000, 1), dtype=np.float16)
# a[0][0] = 22
# import time
#
# start = time.time()
# a.T
# print(time.time() - start)
# print(a[0])
import numba as nb
from numba import cuda
import math
from tqdm import trange

# data =
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


@cuda.jit()
def function(data, traces):
    # idx = cuda.grid(1)
    #
    # # a = 0.0
    # if idx >= 2 ** 32:
    #     return
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
    old_var_data = 0
    old_var_traces = 0
    old_mean_data = 0
    old_mean_traces = 0
    old_cov = 0
    # data[0] ^ idx
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    # 从 idxWithinGrid 开始
    # 每次以整个网格线程总数为跨步数
    for j in range(idx, 2**26, gridStride):
        for i in range(data.shape[0]):
            a = data[i,0]^j
            # print(a)
            new_mean_data = old_mean_data + (a - old_mean_data) / (i + 1)
            new_mean_traces = old_mean_traces + (traces[i] - old_mean_traces) / (i + 1)

            new_var_traces = old_var_traces + (
                    (traces[i] - old_mean_traces) * (traces[i] - new_mean_traces) - old_var_traces) / (
                                     i + 1)
            new_var_data = old_var_data + (
                    (a - old_mean_data) * (a - new_mean_data) - old_var_data) / (
                                   i + 1)
            new_cov = (old_cov * i + (a - old_mean_data) * (traces[i] - new_mean_traces)) / (i + 1)
            # 循环准入
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
            old_mean_traces = new_mean_traces
            old_mean_data = new_mean_data
        print(old_cov / math.sqrt(old_var_traces * old_var_data))
    # cor[idx] = old_cov / math.sqrt(old_var_traces * old_var_data)


data_ = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_labels_AES_HD.npy")
data = data_.astype(np.int32)
# print(data[0,0])
# print(data[0].dtype)
traces_ = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")
traces = np.ascontiguousarray(traces_[:, 0])
A_global_mem = cuda.to_device(data)
B_global_mem = cuda.to_device(traces)
# C_global_mem = cuda.device_array((data.shape[1],traces.shape[1]))
threads_per_block = 1024
blocks_per_grid = int(math.ceil(2 ** 25 / threads_per_block))
for i in trange(2):
    function[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem)
    # 等待所有内核计算完成
    cuda.synchronize()
# C_global_gpu = C_global_mem.copy_to_host()
