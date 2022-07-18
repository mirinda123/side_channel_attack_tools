# 基本示例
def online_mean(old_mean, new_x, N):
    new_mean = old_mean + (new_x - old_mean) / (N + 1)
    return new_mean


def welford(old_var, old_mean, new_x, new_mean):
    new_var = old_var + ((new_x - old_mean) * (new_x - new_mean) - old_var) / (N + 1)
    return new_var
from tqdm import tqdm, trange

# def tt_mean(traces):
#     # 都是按列的
#     t_sum = np.zeros(traces.shape[1])
#     t_num = traces.shape[0]
#     aver_mean = np.zeros(traces.shape[1])
#     n = 0
#     for t in range(t_num):
#         n = n+1
#         t_sum = aver_mean*(n-1)+traces[t,:]
#         aver_mean = t_sum/n
#     return aver_mean
# def tt_var(traces):
#     return tt_mean(traces**2)-(tt_mean(traces))**2
# def calc_corr(a, b):
#     a_avg = sum(a) / len(a)
#     b_avg = sum(b) / len(b)
#     # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
#     cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
#     # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
#     sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
#     corr_factor = cov_ab / sq
#     return corr_factor
# def cor(data,trace1):
#     list = []
#     for t in range(trace1.shape[1]):
#         a = calc_corr(data,trace1[:,t])
#         list.append(a)
#     return np.array(list)
import numpy as np
from numpy import loadtxt


# 读入曲线

# 求矩阵均值按列
def big_mean_func(n, lujing):
    arr = np.load(lujing + r"arrPart0.npy")
    N = arr.shape[1]
    # old_var = np.zeros(N)
    old_mean = np.zeros(N)
    count = 0
    for j in trange(n):

        arr = np.load(lujing + r"arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean = old_mean + (arr[i] - old_mean) / (count + 1)
            # new_var = old_var + ((arr[i] - old_mean) * (arr[i] - new_mean) - old_var) / (count + 1)
            old_mean = new_mean
            # old_var = new_var
            count = count + 1
    return old_mean
def mean_func(traces):

    if traces.ndim == 2:
        old_mean = traces[0]
        # new_mean = np.zeros(traces.shape[1])
        for i in range(traces.shape[0]):

            new_mean = old_mean + (traces[i] - old_mean) / (i + 1)

            old_mean = new_mean
        return old_mean

#lujing不包括文件名
#lujing是r"D:/abc/efg"这样的
#文件名是arrPart
def big_var_func(n,lujing):


    arr = np.load(lujing + r"arrPart0.npy")
    N = arr.shape[1]
    old_var = np.zeros(N)
    old_mean = np.zeros(N)
    count = 0
    for j in trange(n):

        arr = np.load(lujing + r"arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean = old_mean + (arr[i] - old_mean) / (count + 1)
            new_var = old_var + ((arr[i] - old_mean) * (arr[i] - new_mean) - old_var) / (count + 1)
            old_mean = new_mean
            old_var = new_var
            count = count + 1
    return old_var
# 求矩阵方差，按行
def var_func(traces):
    if traces.ndim == 2:
        old_var = np.zeros(traces.shape[1])
        # old_mean = traces[0]
        old_mean = np.zeros(traces.shape[1])
        for i in range(traces.shape[0]):
            new_mean = old_mean + (traces[i] - old_mean) / (i + 1)
            new_var = old_var + ((traces[i] - old_mean) * (traces[i] - new_mean) - old_var) / (i + 1)
            old_mean = new_mean
            old_var = new_var

        return old_var


def big_correlation_func(n, url_trace,url_data):
    arr = np.load(url_trace + r"arrPart0.npy")
    # data = np.load(url_data + r"dataPart0.npy")
    N = arr.shape[1]
    old_cov = np.zeros(N)
    old_mean_data = 0
    old_mean_traces = np.zeros(N)
    old_var_data = 0
    old_var_traces = np.zeros(N)
    count = 0
    for j in trange(n):
        data = np.load(url_data + r"dataPart{0}.npy".format(j))
        arr = np.load(url_trace + r"arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (count + 1)
            new_mean_traces = old_mean_traces + (arr[i] - old_mean_traces) / (count + 1)
            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                    count + 1)
            new_var_traces = old_var_traces + (
                    (arr[i] - old_mean_traces) * (arr[i] - new_mean_traces) - old_var_traces) / (count + 1)
            new_cov = (old_cov * i + (data[i] - old_mean_data) * (arr[i] - new_mean_traces)) / (count + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
            count = count + 1
    return old_cov / np.sqrt(old_var_traces * old_var_data)
# 求一条data和trace每列的协方差
def cov_func(data,traces):
    '''

    :param data: 中间变量
    :param traces: 采样数据
    :return:
    '''
    if data.ndim == 1 and traces.ndim ==2:
        old_cov = np.zeros(traces.shape[1])
        # new_cov = np.zeros((traces.shape[0], traces.shape[1]))
        old_mean_data = np.zeros(len(data))
        # new_mean_data = np.zeros(len(data))
        old_mean_traces = np.zeros(traces.shape[1])
        # new_mean_traces = np.zeros((traces.shape[0], traces.shape[1]))
        count = 0
        for i in range(traces.shape[0]):
            # 初始化
            # 需要的均值
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (i + 1)
            new_mean_traces = old_mean_traces + (traces[i] - old_mean_traces) / (i + 1)
            # 均值的更新
            # 注意这里的i值
            # 协方差的计算
            new_cov = (old_cov*i + (data[i]-old_mean_data)*(traces[i]-new_mean_traces))/(count+1)
            count = count+1
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            # 协方差更新
            old_cov = new_cov
            # print(old_cov[i+1],data[i+1]-old_mean_data[i])
        return old_cov
def correlation_func_dif(data, traces):
    if data.ndim == 1 and traces.ndim == 2:
        old_cov = np.zeros(traces.shape[1])
        old_mean_data = 0
        old_mean_traces = np.zeros(traces.shape[1])
        old_var_data = 0
        old_var_traces = np.zeros(traces.shape[1])
        for i in range(traces.shape[0]):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (i + 1)
            new_mean_traces = old_mean_traces + (traces[i] - old_mean_traces) / (i + 1)
            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                        i + 1)
            new_var_traces = old_var_traces + (
                        (traces[i] - old_mean_traces) * (traces[i] - new_mean_traces) - old_var_traces) / (i + 1)
            new_cov = (old_cov * i + (data[i] - old_mean_data) * (traces[i] - new_mean_traces)) / (i + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
        # print(new_cov[i+1],new_var_traces[i + 1],new_var_data[i + 1])
        return old_cov / np.sqrt(old_var_traces * old_var_data)

def correlation_func_single(data, traces):
    if data.ndim == 1 and traces.ndim == 1:
        old_cov = 0
        old_mean_data = 0
        old_mean_traces = 0
        old_var_data = 0
        old_var_traces = 0
        for i in range(len(data)):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (i + 1)
            new_mean_traces = old_mean_traces + (traces[i] - old_mean_traces) / (i + 1)
            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                        i + 1)
            new_var_traces = old_var_traces + (
                        (traces[i] - old_mean_traces) * (traces[i] - new_mean_traces) - old_var_traces) / (i + 1)
            new_cov = (old_cov * i + (data[i] - old_mean_data) * (traces[i] - new_mean_traces)) / (i + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
        # print(new_cov[i+1],new_var_traces[i + 1],new_var_data[i + 1])
        return old_cov / np.sqrt(old_var_traces * old_var_data)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # data = np.load(r"G:/py+idea/python/side_channel_attack/attackmeasure/traces/SendData.npy")
    # trace = np.load(r"G:/py+idea/python/side_channel_attack/attackmeasure/traces/0x01.npy")
    # print(correlation_func(data, trace))
    # # print(cov_func(data, trace[0]))
    # plt.plot(correlation_func(data, trace))
    # plt.show()
    lujing = r"G:/pycharm/side_channel_attack/attack_method/traces/modu_trace/"

    arr = np.load(r"//attack_method/traces/modu_trace/arrPart0.npy")
    arr1 = np.load(r"//attack_method/traces/modu_trace/arrPart1.npy")
    print(np.mean(arr,axis = 0)+np.mean(arr1,axis = 0))
    print(big_var_func(2,lujing)*2)


    #检查输出路径
    # for j in trange(5):
    #     print(lujing + "arrPart{0}.npy\"" .format(j))
    #
    # #验证路径
    # arrr = np.load(r"G:/pycharm/side_channel_attack/attack_method/traces/modu_trace/arrPart0.npy")
    # print(arrr)
