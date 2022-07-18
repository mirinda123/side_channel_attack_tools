import numpy as np
import math
import time
a = time.time()
from scipy import stats
def function_cpu(data, traces, Result):
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
    Result = np.zeros(2 ** 32, dtype=np.float16)
    for j in range(2 ** 32):
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
            #a1
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
        # a = old_cov / math.sqrt(old_var_traces * old_var_data)
        # if a>=0.01 or a<=-0.01:
        #     print(a)

        # temp是相关系数
        Result[j] = old_cov / math.sqrt(old_var_traces * old_var_data)
    return np.max(Result)
