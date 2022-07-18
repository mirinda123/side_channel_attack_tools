import numpy as np
# import matplotlib as plt

import matplotlib.pyplot as plt

def find_peaks(trace,peak_width,thre):
    if not isinstance(trace, np.ndarray):
        raise TypeError(f"'data' should be a numpy ndarray.")
    length = len(trace)
    initial_trace = []
    for i in range(1,length-1):
        if trace[i-1]<trace[i] and trace[i]>trace[i+1] and trace[i]>thre:
            initial_trace.append(i)
        elif trace[i] == trace[i+1] and trace[i]>thre:
            initial_trace.append(i)
    num = len(initial_trace)
    num2 = 0
    second_trace = initial_trace.copy()
    for j in range(1,num):
        if initial_trace[j]-initial_trace[j-1]<peak_width:
            if initial_trace[j]>initial_trace[j-1]:
                second_trace[j-1] = 0
            else:
                second_trace[j] = 0
            # global num2
            num2 = num2+1
    subnum = num - num2
    second_trace = [i for i in second_trace if i>0]
    peak_index = []
    for i in range(len(second_trace)):
        if i==0:
            index_range = np.array(initial_trace)[np.array(initial_trace)<=second_trace[i]]
        else:
            index_range = np.array(initial_trace)[(np.array(initial_trace)<=second_trace[i])&(np.array(initial_trace)>second_trace[i-1])]
        peak_index.append(index_range[np.argmax(trace[index_range],axis=0)])
    return [subnum,peak_index]
if __name__ == '__main__':

    plt.rcParams['figure.figsize'] = [40,4]
    plt.figure()
    trace = np.load('G:/py+idea/python/sidechannel/pretrace/traces.npy')
    meantrace = np.mean(trace,axis=0)
    # plt.plot(trace.T)
    # print(meantrace)
    # plt.show()
    # plt.plot(meantrace.T)
    # plt.show()
    # find_peaks(meantrace,10,0.5)
    print(find_peaks(meantrace,10,0.05))