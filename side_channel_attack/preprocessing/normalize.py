import numpy as np


# def to_one(traces):
#     if not isinstance(traces, np.ndarray):
#         raise TypeError("'data' should be a numpy ndarray.")
#     new_traces = np.zeros((traces.shape[0],traces.shape[1]), dtype=float)
#     for t in range(traces.shape[0]):
#         a = np.mean(traces[t])
#         s = np.std(traces[t])
#         new_traces[t] = (traces[t]-a)/s
#     return new_traces
# def to_one2(traces):
#     if not isinstance(traces, np.ndarray):
#         raise TypeError("'data' should be a numpy ndarray.")
#     new_traces = np.zeros((traces.shape[0],traces.shape[1]), dtype=float)
#     for t in range(traces.shape[0]):
#         a = np.mean(traces[t])
#         # s = np.std(traces[t])
#         new_traces[t] = (traces[t]-a)
#     return new_traces
# import base_trace.
from tqdm import trange
import numba as nb
# @nb.jit()
def big_temp_func(n,url_traces,file_name):
    arr = np.load(url_traces +file_name+ r"0.npy")
    N = arr.shape[1]
    old_var = np.zeros(N)
    old_mean = np.zeros(N)
    count = 0
    for j in trange(n):
        arr = np.load(url_traces +file_name+ r"{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean = old_mean + (arr[i] - old_mean) / (count + 1)
            new_var = old_var + ((arr[i] - old_mean) * (arr[i] - new_mean) - old_var) / (count + 1)
            old_mean = new_mean
            old_var = new_var
            count = count + 1
    return old_mean,old_var

# @nb.jit()
def to_one(n,url_traces,file_name,save_name):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    temp = big_temp_func(n,url_traces,file_name)
    # print(temp)
    mean_traces = temp[0]
    std_traces = temp[1]
    traces = np.load(url_traces +file_name+ r"0.npy")
    print(traces.shape)
    N = traces.shape[1]
    for j in trange(n):
        traces = np.load(url_traces +file_name+ r"{0}.npy".format(j))
        for i in range(traces.shape[0]):
            traces[i] = (traces[i]-mean_traces)/std_traces
        np.save(url_traces + save_name+ r"{0}.npy".format(j), traces)
        print(traces.shape)

# def to_one2(traces):
#     if not isinstance(traces, np.ndarray):
#         raise TypeError("'data' should be a numpy ndarray.")
#     new_traces = np.zeros((traces.shape[0],traces.shape[1]), dtype=float)
#     for t in range(traces.shape[0]):
#         a = np.mean(traces[t])
#         # s = np.std(traces[t])
#         new_traces[t] = (traces[t]-a)
#     return new_traces
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from preprocessing import filter
    # import random
    import time
    start = time.time()
    from numpy import loadtxt
    to_one(200,r"H:/order2random354besttrace12/",r"passarrPart","yang")
    print(time.time()-start)
