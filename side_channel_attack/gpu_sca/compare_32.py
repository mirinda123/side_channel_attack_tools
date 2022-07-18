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

N = 2 ** 8


# data =


@cuda.jit
def function(data, traces, Result, maxc):
    # idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # gridStride = cuda.gridDim.x * cuda.blockDim.x
    idx = cuda.grid(1)
    if idx >= N:
        return
    HW_table = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
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
                7, 7, 8)
    old_var_data = 0
    old_var_traces = 0
    old_mean_data = 0
    old_mean_traces = 0
    old_cov = 0
    for i in range(data.shape[0]):
        # b = HW_table[0]
        a = data[i, 0] ^ idx
        a0 = (a & 34158280448) >> 24
        a1 = (a & 133299968) >> 16
        a2 = (a & 520448) >> 8
        a3 = (a & 2033)
        if a0 & 128 == 0:
            a0 = (a0 << 1)
        if a0 & 128 == 1:
            a0 = (a0 << 1) ^ 0x1B
        if a1 & 128 == 0:
            a1 = (a1 << 1) ^ a1
        if a1 & 128 == 1:
            a1 = (((a1 << 1) ^ 0x1B) ^ 0x1B)
        a = HW_table[a0 ^ a1 ^ a2 ^ a3]
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
    Result[idx] = old_cov / math.sqrt(old_var_traces * old_var_data)

    cuda.atomic.max(maxc, 0, Result[idx])


def function_cpu(data, traces):
    # idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # gridStride = cuda.gridDim.x * cuda.blockDim.x

    # a = 0.0
    # global j
    HW_table = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
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
                7, 7, 8)
    old_var_data = 0
    old_var_traces = 0
    old_mean_data = 0
    old_mean_traces = 0
    old_cov = 0
    # data[0] ^ idx
    # i 曲线条数
    Result = np.zeros(N, dtype=np.float16)
    for j in range(N):
        for i in range(data.shape[0]):
            # b = HW_table[0]
            a = data[i, 0] ^ j
            a0 = (a & 34158280448) >> 24
            a1 = (a & 133299968) >> 16
            a2 = (a & 520448) >> 8
            a3 = (a & 2033)

            t = a0 ^ a1 ^ a2 ^ a3
            u = a0
            # a0
            if (a0 ^ a1) & 0x80:
                a0 ^= t ^ ((((a0 ^ a1) << 1) ^ 0x1B) & 0xFF)
            else:
                a0 ^= t ^ ((a0 ^ a1) << 1)
            # a1
            if (a1 ^ a2) & 0x80:
                a1 ^= t ^ (((a1 ^ a2) << 1) ^ 0x1B) & 0xFF
            else:
                a1 ^= t ^ ((a1 ^ a2) << 1)
            # a2
            if (a2 ^ a3) & 0x80:
                a2 ^= t ^ (((a2 ^ a3) << 1) ^ 0x1B) & 0xFF
            else:
                a2 ^= t ^ ((a2 ^ a3) << 1)
            # a3
            if (a3 ^ u) & 0x80:
                a2 ^= t ^ (((a3 ^ u) << 1) ^ 0x1B) & 0xFF
            else:
                a2 ^= t ^ ((a3 ^ u) << 1)
            a = HW_table[a0] + HW_table[a1] + HW_table[a2] + HW_table[a3]
            # cuda.atomic.xor(data, [i, 0], idx)
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

        Result[j] = old_cov / math.sqrt(old_var_traces * old_var_data)
    return np.max(Result), Result[0]
from scipy import stats
def identify_function(data,traces):
    a = np.zeros(data.shape[0], dtype=np.float16)
    HW_table = (0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
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
                7, 7, 8)
    for i in range(data.shape[0]):
        a[i] = data[i, 0] ^ 0
        a0 = (a & 34158280448) >> 24
        a1 = (a & 133299968) >> 16
        a2 = (a & 520448) >> 8
        a3 = (a & 2033)

        t = a0 ^ a1 ^ a2 ^ a3
        u = a0
        # a0
        if (a0 ^ a1) & 0x80:
            a0 ^= t ^ ((((a0 ^ a1) << 1) ^ 0x1B) & 0xFF)
        else:
            a0 ^= t ^ ((a0 ^ a1) << 1)
        # a1
        if (a1 ^ a2) & 0x80:
            a1 ^= t ^ (((a1 ^ a2) << 1) ^ 0x1B) & 0xFF
        else:
            a1 ^= t ^ ((a1 ^ a2) << 1)
        # a2
        if (a2 ^ a3) & 0x80:
            a2 ^= t ^ (((a2 ^ a3) << 1) ^ 0x1B) & 0xFF
        else:
            a2 ^= t ^ ((a2 ^ a3) << 1)
        # a3
        if (a3 ^ u) & 0x80:
            a2 ^= t ^ (((a3 ^ u) << 1) ^ 0x1B) & 0xFF
        else:
            a2 ^= t ^ ((a3 ^ u) << 1)
        a[i] = HW_table[a0] + HW_table[a1] + HW_table[a2] + HW_table[a3]
    return (stats.pearsonr(a,traces))
result_ = np.zeros(N, dtype=np.float32)
maxc = np.zeros(1, dtype=np.float32)
# print(data[0,0])
# print(data[0].dtype)
data_ = np.load(r"attack_labels_AES_HD.npy")
data = data_.astype(np.int32)
# print(data[0,0])
# print(data[0].dtype)
traces_ = np.load(r"attack_traces_AES_HD.npy")

# ???
traces = np.ascontiguousarray(traces_[:, 0])
A_global_mem = cuda.to_device(data)
B_global_mem = cuda.to_device(traces)
Result = cuda.to_device(result_)
# C_global_mem = cuda.device_array((data.shape[1],traces.shape[1]))

# 最大开1024
threads_per_block = 1024
blocks_per_grid = int(math.ceil(N / threads_per_block))
for i in trange(2):
    function[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, Result, maxc)
    # 等待所有内核计算完成
    cuda.synchronize()
    # print(a)
result123 = Result.copy_to_host()
# print(result123.shape)
print("maxc[0]",maxc[0])
print("identify_function(data,traces)",identify_function(data,traces))
print("function_cpu(data, traces)",function_cpu(data, traces))
# C_global_gpu = C_global_mem.copy_to_host()
