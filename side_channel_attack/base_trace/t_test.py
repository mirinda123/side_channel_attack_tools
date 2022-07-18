import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import time
import numba as nb




@nb.jit
def t_test(n, url_trace,filename):
    '''
    :instrctions:even  and  odd 间或采集
    :param n: blocks of traces
    :param N: sample numbers
    :return: result
    '''
    arr = np.load(url_trace + filename+"arrPart0.npy")
    count = 0
    N = arr.shape[1]
    old_var_even = np.zeros(N)
    old_mean_even = np.zeros(N)
    old_var_odd = np.zeros(N)
    old_mean_odd = np.zeros(N)
    oddcount = 0
    evencount = 0
    for j in trange(n):
        arr = np.load(url_trace + filename + "arrPart{0}.npy".format(j))
        for i in range(arr.shape[0]):
            if count % 2 == 0:
                new_mean = old_mean_even + (arr[i] - old_mean_even) / (evencount + 1)
                new_var = old_var_even + ((arr[i] - old_mean_even) * (arr[i] - new_mean) - old_var_even) / (
                            evencount + 1)
                old_mean_even = new_mean
                old_var_even = new_var
                evencount += 1
                # print(evencount)
                count = count + 1
            else:
                new_mean = old_mean_odd + (arr[i] - old_mean_odd) / (oddcount + 1)
                new_var = old_var_odd + ((arr[i] - old_mean_odd) * (arr[i] - new_mean) - old_var_odd) / (oddcount + 1)
                old_mean_odd = new_mean
                old_var_odd = new_var
                oddcount += 1
                count = count + 1
    temp1 = old_mean_even - old_mean_odd
    temp2 = (old_var_even / evencount) + (old_var_odd / oddcount)
    test_result = temp1 / np.sqrt(temp2)
    return test_result


def plt_t_test(all_kinds_of_results):
    plt.rcParams['figure.figsize'] = (12.0, 7.0)
    f, ax = plt.subplots(1, 1)
    plt.plot(t_test(1, url_trace))
    ax.axhline(y=4.5, ls='--', c='red', linewidth=2)
    ax.axhline(y=-4.5, ls='--', c='red', linewidth=2)
    plt.show()



if __name__ == '__main__':

    block_num = 200
    #filename = "aes_allzero_PICO5000_125MHZ"
    filename = "after_process_"
    #url_trace = r'D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/'
    url_trace = r'E:/7yue16/'
    plt.rcParams['figure.figsize'] = (12.0, 7.0)
    f, ax = plt.subplots(1, 1)
    ax.axhline(y=4.5, ls='--', c='red', linewidth=2)
    ax.axhline(y=-4.5, ls='--', c='red', linewidth=2)
    result = t_test(block_num, url_trace,filename)
    np.save(r'D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\tutorials\ttest_result_blinky.npy', result )
    # result = np.load(r'G:/side_channel_attack/result_io_redo_share5_edit_synchronized.npy')

    plt.plot(result)
    # plt.axvline(x=0)
    # plt.axvline(x=119)
    # plt.axvline(x=155)
    # plt.axvline(x=192)
    # plt.axvline(x=229)
    # plt.axvline(x=268)
    # plt.axvline(x=293)
    # plt.axvline(x=333)
    # plt.axvline(x=352)
    # plt.axvline(x=392)
    # plt.axvline(x=416)
    # plt.axvline(x=645)
    plt.show()

    start = time.time()
    # transform(400)


    print(time.time()-start)
