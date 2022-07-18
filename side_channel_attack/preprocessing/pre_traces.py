import numba
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm, trange
from numba import cuda
import numba as nb

np.seterr(divide='ignore', invalid='ignore')


# import random
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


@nb.jit
def to_one2(traces):

    # 判断是否是一个类型isinstance 不能用在numba里面
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1]), dtype=float)
    a = np.mean(traces, axis=0)
    for t in range(traces.shape[0]):
        # a = np.mean(traces, axis=0)
        new_traces[t] = (traces[t] - a)
    return new_traces


@nb.jit
def fast_lowpass(traces, a):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'trace' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1]), dtype=float)
    # point_list = []
    for t in range(traces.shape[0]):
        for i in range(0, traces.shape[1]):
            if i == 0:
                new_traces[t][0] = traces[t][0]
            else:
                new_traces[t][i] = (a * traces[t][i - 1] + traces[t][i]) / (1 + a)
    return new_traces


# @cuda.jit
def LowPass(trace, frequency, cutoff, axis=-1, precision='float32'):
    # if not isinstance(trace, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    # if not isinstance(frequency, int) and not isinstance(frequency, float):
    #     raise TypeError("'frequency' should be an of int or float type.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    b, a = signal.butter(3, 2 * cutoff / frequency, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, trace)
    return filtedData


@cuda.jit
def fast_lowpass_wanggeyibu(traces, a, new_traces):
    t = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    for j in range(t, traces.shape[0], gridStride):
        for i in range(0, traces.shape[1]):
            if i == 0:
                new_traces[t][0] = traces[t][0]
            else:
                new_traces[t][i] = (a * traces[t][i - 1] + traces[t][i]) / (1 + a)


@nb.jit
def moving_resamples(traces, k):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")

    # 默认数据类型是float64
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


def moving_resamples2(traces, k):
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1] // k), dtype=float)
    if k >= traces.shape[1]:
        raise ValueError('window is too bigooo!')
    n = traces.shape[1]
    for t in range(traces.shape[0]):
        list = []
        # //地板除
        for i in range(n // k):
            # sum = 0
            # for j in range(i * k, i * k + k):
            #     sum = traces[t][j] + sum
            # print(traces[t][i*k:i*(k+1)-1])
            a = np.mean(traces[t][i * k:i * (k + 1) - 1])
            list.append(a)
        new_traces[t] = np.array(list)
    return new_traces


def cut_func(traces, min=None, max=None):
    if traces.ndim == 1:
        new_traces = traces[min:max]
        return new_traces
    if traces.ndim == 2:
        new_traces = np.zeros((traces.shape[0], max - min))
        for i in range(traces.shape[0]):
            new_traces[i] = traces[i][min:max]
        return new_traces
#200   D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\tutorials\  after_process_
def all_files_to_one(n,url,filename):

    # n是块数目
    arr = np.load(url+filename + "arrPart0.npy")
    M = arr.shape[1]
    old_mean = np.zeros(M)
    count = 0
    for j in trange(n):
        arr = np.load(url + filename + "arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            new_mean = old_mean + (arr[i] - old_mean) / (count + 1)
            old_mean = new_mean
            count = count + 1



    for j in trange(n):
        new_traces = np.zeros((arr.shape[0], arr.shape[1]), dtype=float)
        arr = np.load(url + filename + "arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            # a = np.mean(traces, axis=0)
            new_traces[i] = (arr[i] - old_mean)

        np.save(url + filename + "arrPart{0}.npy".format(j),new_traces)
if __name__ == '__main__':


    block_num = 200
    # 对曲线进行处理  大量多组曲线
    for j in trange(block_num):

        tracepass1 = np.load(r"D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\tutorials\skinny_random_PICO5000_125MHZarrPart{0}.npy".format(j))

        # 对AES，moving的系数是20效果比较好

        # 对PRESENT,moving的系数是30比较好

        # 对SKINNY,moving的系数是33比较好
        #scaler = preprocessing.StandardScalser()
        #tracepass1 = scaler.fit_transform(tracepass1)

        # wangbohan的代码，50效果还可以

        #tracepass1 = moving_resamples(tracepass1, 4)
        tracepass1 = fast_lowpass(tracepass1,10)
        tracepass1 = to_one2(tracepass1)

        #lowpass的第二个参数，是采样率

        #tracepass1 = LowPass(tracepass1, 40e6, 0.8e6)
        #=tracepass1 = tracepass1 / 10000
        #scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        #tracepass1 = scaler.fit_transform(tracepass1)
        #plt.plot(tracepass1[256])
        #plt.show()
        #print(tracepass1)

        #np.save(r'D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\tutorials\after_process_arrPart{0}.npy'.format(j), tracepass1)
        np.save(r"E:\7yue16\after_process_arrPart{0}.npy".format(j),tracepass1)
        # tracepass = fast_lowpass(arr,50)
        # np.save('D:/1222order1EditAlphaSboxMixAlphais1trace1trace2/yiwanarrPart{0}.npy'.format(j), tracepass)

    # url  = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/"
    # filename ="after_process_"
    # 下面这个纵向归一化没经过验证，不知道对不对
    # all_files_to_one(block_num,url,filename)