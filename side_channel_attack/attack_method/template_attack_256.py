import numpy as np
from side_channel_attack.base_trace.statistics_tools import mean_func, cov_func
from scipy.stats import multivariate_normal
from base_trace.data_resolve import HW


# model_data应该为对应trace的中间值，故事先用中间值函数计算出来
#  改成对256种密钥进行建模
def find_pois(model_data, model_traces, attack_data, attack_traces, num_pois, poi_space):
    """
    :param model_data: 用来建模的曲线集对应的中间值
    :param model_traces: 用来建模的曲线集
    :param attack_data: 模板攻击用到的曲线对应的 中间值
    :param attack_traces: 模板攻击用到的曲线
    :param num_pois: 设定好一条曲线有几个泄漏点
    :param poi_space: 泄漏点周边范围内设置为0
    :return: 攻击出来的值
    """
    temp_trace_label = [[] for _ in range(256)]
    temp_means = np.zeros((256, model_traces.shape[1]))
    temp_sum_diff = np.zeros(model_traces.shape[1])
    poi_lists = []
    # Fill them up
    for i in range(model_traces.shape[0]):
        label_number = int(model_data[i])
        temp_trace_label[label_number].append(model_traces[i])
    # Switch to numpy arrays
    temp_trace_label = [np.array(temp_trace_label[label_number]) for label_number in range(256)]
    for i in range(256):
        temp_means[i] = mean_func(temp_trace_label[i])
        for j in range(i):
            # 向量
            temp_sum_diff += np.abs(temp_means[i] - temp_means[j])
    # 5: Find POIs
    for i in range(num_pois):
        # Find the max
        next_poi = temp_sum_diff.argmax()
        poi_lists.append(next_poi)
        # Make sure we don't pick a nearby value
        poi_min = max(0, next_poi - poi_space)
        poi_max = min(next_poi + poi_space, len(temp_sum_diff))
        for j in range(poi_min, poi_max):
            temp_sum_diff[j] = 0
    mean_matrix = np.zeros((256, num_pois))
    cov_matrix = np.zeros((256, num_pois, num_pois))
    for label_number in range(256):
        for i in range(num_pois):
            # Fill in mean
            mean_matrix[label_number][i] = temp_means[label_number][poi_lists[i]]
            for j in range(num_pois):
                x = temp_trace_label[label_number][:, poi_lists[i]]
                y = temp_trace_label[label_number][:, poi_lists[j]]
                cov_matrix[label_number, i, j] = cov_func(x, y)

    P_k = np.zeros(256)
    for j in range(attack_traces.shape[0]):
        a = [attack_traces[j][poi_lists[i]] for i in range(len(poi_lists))]
        # Test each key
        for k in range(256):
            # 此处中间值为data
            label_number = attack_data[j]
            rv = multivariate_normal(mean_matrix[label_number], cov_matrix[label_number])
            p_kj = rv.logpdf(a)
            P_k[k] += p_kj
    guess = P_k.argsort()[-1]
    return hex(guess)


if __name__ == '__main__':
    data1 = HW(np.load(r"G:/py+idea/python/sidechannel/attackmeasure/traces/SendData.npy"))

    # print(HW(data1))
    trace = np.load(r"G:/py+idea/python/sidechannel/pretrace/mtraces/m0x01evenodd.npy")
    print(find_pois(data1, trace, 5, 4))
