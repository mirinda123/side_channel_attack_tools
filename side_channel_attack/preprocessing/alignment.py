from scipy import signal
import numpy as np
from fastdtw import fastdtw
# from base import get_trace_sample
from find_peaks import *
import scipy.stats as stats
import matplotlib.pyplot as plt
# from base_trace.statistics_tools import var_func,mean_func,correlation_func

# # 在噪声加入的情况下表现不好，但可以同步曲线  目前是两个曲线要等长
# def align_traces(ref, trace):
#     N = trace.shape[1]
#     r = 1
#     aref = np.array(list(ref))
#     atrace = np.array(list(trace))
#     dist, path = fastdtw(aref, atrace, radius=r, dist=None)
#     px = [x for x, y in path]
#     py = [y for x, y in path]
#     n = [0] * N
#     s = [0.0] * N
#     for x, y in path:
#         s[x] += trace[y]
#         n[x] += 1
#
#     ret = [s[i] / n[i] for i in range(N)]
#     return ret

# remove traces###########
# def corelation_align(traces,initialtrace,reftrace,leftmin, rightmax):
#     '''
#
#     :param traces: traces
#     :param initialtrace: set reftrace
#     :param reftrace: orther trace need to align
#     :param leftmin: refpattern leftmin value
#     :param rightmax: refpattern rightmax value
#     :return:
#     '''
#     # initialtrace为参照曲线   othertraces为其他曲线
#     if not isinstance(initialtrace, np.ndarray):
#         raise TypeError("'data' should be a numpy ndarray.")
#     if not isinstance(reftrace, np.ndarray):
#         raise TypeError("'data' should be a numpy ndarray.")
#     # ref因该为另一曲线的一部分
#     ref = reftrace[leftmin:rightmax]
#     t = len(initialtrace)
#     r = len(ref)
#     correct_list = []
#     # for i in range(0,t-r+1):
#     #     tr_correct = correlation_func(traces[i:i+r],reftrace)
#     for i in range(0,t-r+1):
#         tr_correct = correlation_func(initialtrace[i:i+r],ref)
#         correct_list.append(tr_correct)
#     # 从左到右遇到的第一个极大相关性值   因为其中一条曲线为所有采样点，故np.argmax即可代表两条曲线偏差
#     m = np.argmax(correct_list)
#     n = m-leftmin
#     # 初始化对其之后的曲线
#     newref_trace = np.zeros(len(reftrace))
#     # 因为用到均值填充
#     mean_trace = mean_func(traces)
#     if n==0:
#         newref_trace = reftrace
#     if n>0:
#         newref_trace[0:n-1] = mean_trace[0:n-1]
#         newref_trace[n:len(reftrace)] = reftrace[0:len(reftrace)-n]
#     if n<0:
#         newref_trace[len(reftrace)+n+1:len(reftrace)] = mean_trace[len(reftrace)+n+1:len(reftrace)]
#         newref_trace[0:len(reftrace)+n] = reftrace[-n:len(reftrace)]
#     return newref_trace
def corelation_align_yu(traces,initialtrace,reftrace,leftmin, rightmax,field,peak_value):
    '''

    :param traces: traces
    :param initialtrace: set reftrace
    :param reftrace: orther trace need to align
    :param leftmin: refpattern leftmin value
    :param rightmax: refpattern rightmax value
    :return:
    '''
    # initialtrace为参照曲线   othertraces为其他曲线
    if not isinstance(initialtrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if not isinstance(reftrace, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    # ref因该为另一曲线的一部分
    ref = reftrace[leftmin:rightmax]
    # t = len(initialtrace)
    r = len(ref)
    correct_list = []
    # for i in range(0,t-r+1):
    #     tr_correct = correlation_func(traces[i:i+r],reftrace)
    for i in range(leftmin-field,rightmax+field):
        tr_correct = stats.pearsonr(initialtrace[i:i+r],ref)
        correct_list.append(tr_correct[0])
    # 从左到右遇到的第一个极大相关性值   因为其中一条曲线为所有采样点，故np.argmax即可代表两条曲线偏差
    print(correct_list)
    print(np.max(correct_list))
    print(np.argmax(correct_list)+leftmin-field)
    N = len(reftrace)
    newref_trace = np.zeros(N)
    mean_trace = np.mean(traces)
    if (np.max(correct_list)<peak_value):
        newref_trace = reftrace
        return newref_trace
    else:

        m = np.argmax(correct_list)+leftmin-field
        n = m-leftmin

        if n==0:
            newref_trace = initialtrace
        if n < 0:
            # newref_trace[0:n-1] = mean_trace[0:n-1]
            # newref_trace[n:N] = initialtrace[0:N-n]
            newref_trace[0:-n - 1] = mean_trace[0:-n - 1]
            newref_trace[-n:N] = initialtrace[0:N + n]
            # newref_trace = reftrace
        if n > 0:
            # newref_trace[N+n+1:N] = mean_trace[N+n+1:N]
            # newref_trace[0:N+n] = initialtrace[-n:N]
            newref_trace[N - n + 1:N] = mean_trace[N - n + 1:N]
            newref_trace[0:N - n] = initialtrace[n:N]
            # newref_trace = reftrace
        return newref_trace

if __name__ == '__main__':
    trace1 = np.load(r"G:/side_channel_attack/side_channel_attack/attack_method/traces/label_data.npy")
    # trace1 = np.load(r"G:/py+idea/python/sidechannel/attackmeasure/antrace.npy")
    t1 = trace1[0]
    t2 = trace1[1][250:500]
    print(corelation_align_yu(trace1,trace1[0],trace1[1],100,150,5,0.1))
    # plt.plot(corelation_align(trace1,trace1[0],trace1[1],250,500))
    # plt.show()




