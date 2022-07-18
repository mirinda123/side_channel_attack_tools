import numpy
import numpy as np
from numpy import *
'''带扩展功能，比如fftshift    fftabs    z中心以及相位'''
# 取振幅  归一化  折半取单边
def fft_traces(traces,rangemin = None,rangemax = None):
    '''

    :param traces: 实数fft  只考虑振幅不考虑相位
    :return:
    '''
    if not isinstance(traces, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if rangemin != None or rangemax != None:
        if traces.ndim == 1:
            N = len(traces)
            # 归一化
            return np.array(1.0 / N * np.fft.fftshift(np.fft.fft(traces)))
        if traces.ndim == 2:
            fft_list = []
            for i in range(traces.shape[0]):
                N = traces.shape[1]
                fft_list.append((np.fft.fft(traces[i])))
            return np.array(fft_list)
    else:
        if traces.ndim == 1:
            N = len(traces)
            # 归一化
            return np.array(1.0 / N * np.fft.fftshift(np.fft.fft(traces)))
        if traces.ndim == 2:
            fft_list = []
            for i in range(traces.shape[0]):
                N = traces.shape[1]
                fft_list.append((np.fft.fft(traces[i])))
            return np.array(fft_list)

def ifft_traces(traces):

    if not isinstance(traces, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if traces.ndim == 1:
        N = len(traces)
        return np.array(1.0 / N * np.fft.ifftshift(np.fft.ifft(traces)))
    if traces.ndim == 2:
        ifft_list = []
        for i in range(traces.shape[0]):
            ifft_list.append((np.fft.ifft(traces[i])))
        return np.array(ifft_list)

if __name__ == '__main__':
    pass
    # f = np.fft.fft2(n,axes= - 1)
    # f = np.fft.fft2(n)
    #
    # # fshift = np.fft.fftshift(f)
    # # freq = np.fft.fftfreq(n.size, d=1)
    # # res = np.log(np.abs(fshift))
    #
    #
    # plt.plot(f[0])
    # plt.show()
        # pass

    # print(f.ndim)
    # a = np.mgrid[:5, :5][0]

    # b = np.array(range(8))
    # m = fft_oper()
    # import matplotlib.pyplot as plt
    # cc = np.fft.fft(n)
    # # nn = m.ifft_traces(cc)
    # # print(n.shape)
    # # print(m.fft_traces(n))
    # # print(fft_oper.fft_traces(n))
    # print(cc)
    # plt.plot(cc.T)
    # plt.show()