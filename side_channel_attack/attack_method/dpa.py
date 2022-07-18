import numpy as np
from tqdm import trange
def mean_func(traces):
    if traces.ndim == 2:
        old_mean = traces[0]
        # new_mean = np.zeros(traces.shape[1])
        for i in range(traces.shape[0]):
            new_mean = old_mean + (traces[i] - old_mean) / (i + 1)
            old_mean = new_mean
        return old_mean
def dpa_attack(n,url_data, url_trace, subkey_num):
    '''

    :param data: intermediate value  对应一个子密钥
    :param traces: traces
    :param subkey_num: subkey numbers

    :return:
    '''
    data = np.load(url_data + r"arrPart0.npy")
    mean_diffs = np.zeros(data.shape[1])
    key_guess = []
    # 逻辑有点问题
    for subkey in range(subkey_num):
        # 列举所有密钥猜测情况
        for kguess in range(data.shape[1]):
            all_one_list = []
            all_zero_list = []
            for j in trange(n):
                traces = np.load(url_trace + r"arrPart{0}.npy".format(j))
                data = np.load(url_data + r"arrPart{0}.npy".format(j))
                # for i in range(arr.shape[0]):
                one_list = []
                zero_list = []
                for i in range(traces.shape[0]):
                    if (data[i][kguess] & 1):
                        one_list.append(traces[i])
                    else:
                        zero_list.append(traces[i])
                one_avg = mean_func(np.asarray(one_list))
                zero_avg = mean_func(np.asarray(zero_list))
                all_zero_list.append(zero_avg)
                all_one_list.append(one_avg)
            all_one_avg = mean_func(np.asarray(all_one_list))
            all_zero_avg = mean_func(np.asarray(all_zero_list))
            mean_diffs[kguess] = np.max(abs(all_one_avg - all_zero_avg))
        #     选取差别最大值处为该子密钥
        guess = np.argsort(mean_diffs)[-1]
        key_guess.append(guess)
    return key_guess
if __name__ == '__main__':
    pass
