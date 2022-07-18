import numpy as np
from  matplotlib import pyplot as plt
import scared
from numpy import loadtxt
from tqdm import tqdm, trange
# i_list = []此处应是数组   np.array
# def snr(data,trace):
def snr(n, url_trace,url_data):
    # if trace.shape[0] != data.shape[0]:
    #     raise ValueError("they should have same imediate value.")
    # if not isinstance(trace, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    # if not isinstance(data, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    trace = np.load(url_trace + r"arrPart0.npy")
    a = np.zeros((256,trace.shape[1]),dtype=float)
    b = np.zeros((256,trace.shape[1]),dtype=float)
    c = np.zeros((256,trace.shape[1]),dtype=float)

    mean_std_temp = np.zeros(trace.shape[1], dtype=float)
    mean_std_temp1 = np.zeros(trace.shape[1], dtype=float)
    mean_std_temp2 = np.zeros(trace.shape[1], dtype=float)
    std_mean_temp3 = np.zeros(trace.shape[1], dtype=float)
    m = 0
    for i in range(256):
        temp1 = np.zeros(trace.shape[1], dtype=float)
        temp2 = np.zeros(trace.shape[1], dtype=float)
        temp3 = np.zeros(trace.shape[1], dtype=float)

        count = 0
        # 在这里添加曲线块数循环
        for j in trange(n):
            data = np.load(url_data + r"dataPart{0}.npy".format(j))
            trace = np.load(url_trace + r"arrPart{0}.npy".format(j))
            for t in range(trace.shape[0]):
                if (data[t] == i):
                    # 这个时候来的是trace_array[tnum]曲线
                    # 需要记录下，x,x**2
                    # 考虑精度，来一次除一次
                    count += 1
                    t_sum = temp1*(count-1) + trace[t, :]
                    t2_sum = temp2*(count-1)+(trace**2)[t, :]
                    # 均值
                    temp1 = t_sum/count
                    # 平方的均值
                    temp2 = t2_sum/count
                    temp3 = temp2 - temp1 ** 2
                else:
                    continue
        a[i] = temp1
        b[i] = temp1**2
        m += 1
        a_sum = mean_std_temp1*(m-1)+temp1
        b_sum = mean_std_temp2*(m-1)+temp1**2
        mean_std_temp1 = a_sum/m
        mean_std_temp2 = b_sum/m
        mean_std_temp = mean_std_temp2-mean_std_temp1 ** 2
        c[i] = temp3
        c_sum = std_mean_temp3*(m-1) + temp3
        std_mean_temp3 = c_sum/m
    return mean_std_temp/std_mean_temp3
if __name__ == '__main__':

    # print(snr(data,trace))
    # plt.plot(data)
    # plt.plot(trace)
    # 太蠢了，直接传参的时候限定范围就好，函数不用改
    # data = []
    loadData = loadtxt(r"D:\traces\sendData.csv", delimiter=',',dtype="int")
    data = loadData
    loadTrace = loadtxt(r"D:\traces\trace0.csv",delimiter=',')
    # 循环读取后面的trace
    for i in range(1, 2000):
        newLoadTrace = loadtxt(
            r"D:\traces\trace{0}.csv".format(i),
            delimiter=',')
        # 拼接为矩阵
        loadTrace = np.vstack([loadTrace, newLoadTrace])

    # 最终得到的loadTrace是一个二维矩阵
    trace = scared.signal_processing.filters.butterworth(loadTrace,30e6,20e5)

    plt.plot(snr(data,trace[:,0:200]))

    plt.show()






