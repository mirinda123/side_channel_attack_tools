import numpy as np
from numba import cuda, float32,jit,vectorize
import time
import numba
from tqdm import tqdm, trange

# def moving_average(self,trace,window):
#     if not isinstance(trace, np.ndarray):
#         raise TypeError(f"'data' should be a numpy ndarray, not {type(data)}.")
#     if window >= len(trace):
#         raise ValueError('window is too bigooo!')
#     n = len(trace)
#     list = []
#     for i in range(n-window):
#         sum = 0
#         for j in range(i,i+window):
#             sum = trace[j]+sum
#         a = sum/window
#         list.append(a)
#     return list
# @vectorize(["float64(float64, float64)"], target='cuda')
def big_moving_average(n,url):
    for j in trange(n):
        arr = np.load(url + r"arrPart{0}.npy".format(j))
        trace_moving_average = moving_average(arr, 5)
        np.save(url + r"moving_arrPart{0}.npy".format(j), trace_moving_average)

@numba.jit
def moving_average(traces, window):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    new_traces = []
    for t in range(traces.shape[0]):
        list = []
        new_traces.append(list)
        for i in range(traces.shape[1] - window):
            sum = 0
            for j in range(i, i + window):
                sum = traces[t][j] + sum
            a = sum / window
            list.append(a)
    return np.array(new_traces)
def big_moving_resamples(n,url):
    for j in trange(n):
        arr = np.load(url + r"arrPart{0}.npy".format(j))
        trace_moving_resamples = moving_average(arr, 5)
        np.save(url + r"resamples_arrPart{0}.npy".format(j), trace_moving_resamples)

@numba.jit
def moving_resamples(traces, k):
    if not isinstance(traces, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
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


if __name__ == '__main__':
    start_cpu = time.time()
    trace = np.load(r"/attack_method/traces/finalBeforeOnlineSboxusingPico.npy")
    trace1 = moving_average[2,2](trace, 400)
    end_cpu = time.time()
    time_cpu = (end_cpu - start_cpu)
    print("CPU process time: " + str(time_cpu))


