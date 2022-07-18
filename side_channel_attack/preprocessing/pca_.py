

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def PCA_(traces, k: int):
    '''

    :param traces:
    :param k: 主成分数量
    :return:
    '''
    if not isinstance(traces, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if k <= 0:
        raise ValueError("k can't <0")
    if k >= traces.shape[1]:
        raise ValueError("k is toooo big!!!")
    pca = PCA(n_components=k)
    pca.fit(traces)
    traces_new = pca.transform(traces)

    return np.array(traces_new)

# def inv_pca()

if __name__ == '__main__':
    n = np.load('G:/py+idea/python/sidechannel/pretrace/aes_traces.npy')

    plt.plot(n.T)
    plt.show()
    print(PCA_(n, 1))
    print(n.shape)
