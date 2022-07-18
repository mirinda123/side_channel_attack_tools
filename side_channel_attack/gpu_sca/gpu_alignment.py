import numpy as np
import scipy.stats as stats
from tqdm   import trange
import numba as nb
from numba import cuda
import time
# @nb.jit()
# correct_liat 大小既是left-field----right+filed
@cuda.jit
def caclate_function(initial,ref,correct_list):
    idx = cuda.grid(1)
    if idx >= initial.shape[0]:
        return
    r = len(ref)
    # correct_list = []
    for i in range(initial.shape[1]):
        old_cov = 0
        old_mean_ref = 0
        old_mean_initial = 0
        old_var_ref = 0
        old_var_initial = 0
        temp = 0
        for j in range(i,i+r):
            new_mean_ref = old_mean_ref + (ref[temp] - old_mean_ref) / (temp + 1)
            new_mean_initial = old_mean_initial + (initial[idx][j] - old_mean_initial) / (temp + 1)
            new_var_ref = old_var_ref + ((ref[temp] - old_mean_ref) * (ref[temp] - new_mean_ref) - old_var_ref) / (
                    temp + 1)
            new_var_initial = old_var_initial + (
                    (initial[idx][j] - old_mean_initial) * (initial[idx][j] - new_mean_initial) - old_var_initial) / (temp + 1)
            new_cov = (old_cov * temp + (ref[temp] - old_mean_ref) * (initial[idx][j] - new_mean_initial)) / (temp + 1)
            old_mean_ref = new_mean_ref
            old_mean_initial = new_mean_initial
            old_cov = new_cov
            old_var_initial = new_var_initial
            old_var_ref = new_var_ref
            temp = temp+1
        correct_list[idx][i] = old_cov / math.sqrt(old_var_ref * old_var_initial)

def corelation_align_yu(initialtrace, reftrace, leftmin, rightmax, field, peak_value):
    if not isinstance(initialtrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(reftrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    # ref因该为另一曲线的一部分
    ref = reftrace[leftmin:rightmax]
    r = len(ref)
    mean_trace = np.mean(initialtrace, axis=0)
    for j in trange(initialtrace.shape[0]):
        correct_list = []
        for i in range(leftmin - field, rightmax + field):
            tr_correct = stats.pearsonr(ref, initialtrace[j][i:i + r])
            correct_list.append(tr_correct[0])
        N = len(reftrace)
        # print(correct_list)
        correct_list = np.array(correct_list)
        if (np.max(abs(correct_list)) < peak_value):
            initialtrace[j] = reftrace
        else:
            m = np.argmax(abs(correct_list)) + leftmin - field
            # print("mmmjjjj", m, j)
            n = m - leftmin
            if n == 0:
                initialtrace[j] = initialtrace[j]
            if n < 0:
                initialtrace[j][0:-n - 1] = mean_trace[0:-n - 1]
                initialtrace[j][-n:N] = initialtrace[j][0:N + n]
            if n > 0:
                # print(initialtrace[j][N - n + 1:N])
                # print( mean_trace[0:-n - 1])
                initialtrace[j][N - n + 1:N] = mean_trace[N - n + 1:N]
                initialtrace[j][0:N - n] = initialtrace[j][n:N]
    return initialtrace
import math
def corelation_align_gpu(initialtrace,reftrace,leftmin, rightmax,field,peak_value):
    if not isinstance(initialtrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(reftrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    # ref因该为另一曲线的一部分
    ref = reftrace[leftmin:rightmax]
    initial = np.ascontiguousarray(initialtrace[:,leftmin-field:rightmax+field])
    start = time.time()
    gpu_ref = cuda.to_device(ref)
    gpu_initial = cuda.to_device(initial)
    gpu_correct_list = cuda.device_array((initial.shape[0],initial.shape[1]))
    threads_per_block = 16
    blocks_per_grid = int(math.ceil(initial.shape[0] / threads_per_block))
    # start = time.time()
    caclate_function[blocks_per_grid, threads_per_block](gpu_initial, gpu_ref,gpu_correct_list)
    # 等待所有内核计算完成
    cuda.synchronize()
    N = len(reftrace)
    mean_trace = np.mean(initialtrace,axis=0)
    correct_list = gpu_correct_list.copy_to_host()
    correct_list = np.array(correct_list)
    for i in trange(correct_list.shape[0]):
    # for j in trange(initialtrace.shape[0]):
        if (np.max(abs(correct_list))<peak_value):
            initialtrace[i] = reftrace
        else:
            m = np.argmax(abs(correct_list[i]))+leftmin-field
            # print(correct_list[i])
            # print("mmmjjjj",m)
            n = m-leftmin
            if abs(n)==0:
                initialtrace[i] = initialtrace[i]
            if n < 0:
                initialtrace[i][0:-n - 1] = mean_trace[0:-n - 1]
                initialtrace[i][-n:N] = initialtrace[i][0:N + n]
            if n > 0:
                # print(initialtrace[j][N - n + 1:N])
                # print( mean_trace[0:-n - 1])
                initialtrace[i][N - n + 1:N] = mean_trace[N - n + 1:N]
                initialtrace[i][0:N - n] = initialtrace[i][n:N]
    print("compare",time.time()-start)
    return initialtrace


        # print(np.argmax(correct_list[i])+leftmin-field)


if __name__ == '__main__':
    trace1 = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/attack_traces_AES_HD.npy")[0:10]
    # trace1 = np.load(r"G:/py+idea/python/sidechannel/attackmeasure/antrace.npy")
    print(trace1)
    # print(corelation_align_yu(trace1,trace1[1],100,150,50,0.00001))
    print(corelation_align_gpu(trace1, trace1[1], 100, 150, 5, 0.00001))
    # plt.plot(corelation_align(trace1,trace1[0],trace1[1],250,500))
    # plt.show()




