import numpy as np
from Crypto.Cipher import AES

import numpy as np
# 有时间加一个类型限制   整数类型哦
def _hamming_weight(intermediate_value):
    '''

    :param intermediate_value: 一个中间值
    :return: 中间值的汉明重量
    '''
    m = 0
    while intermediate_value != 0:
        # print(type(intermediate_value))
        m=m+(intermediate_value & 1)
        intermediate_value >>= 1
    return m
def _LSB(intermediate_value):
    '''

    :param intermediate_value: 一个中间值
    :return: 返回最低有效位
    '''
    m = 0
    while intermediate_value != 0:
        m = intermediate_value & 1
    return m
def HW(data):
    '''

    :param data: 一组中间值
    :return: 一组中间值的汉明重量
    '''
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if data.ndim == 1:
        new_data = np.zeros(len(data), dtype=float)
        for i in range(len(data)):
            new_data[i] = _hamming_weight(int(data[i]))
        return new_data
    if data.ndim == 2:
        new_data = np.zeros((data.shape[0],data.shape[1]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_data[i][j] = _hamming_weight(int(data[i][j]))
        return new_data
def LSB(data):
    """
    :param data:同上
    :return:
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if data.ndim == 1:
        new_data = np.zeros(len(data), dtype=float)
        for i in range(len(data)):
            new_data[i] = _LSB(data[i])
        return new_data
    if data.ndim == 2:
        new_data = np.zeros((data.shape[0],data.shape[1]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_data[i][j] = _LSB(data[i][j])
        return new_data
# print(HW(np.array(255)))



def _hamming_weight(intermediate_value):
    '''

    :param intermediate_value: 一个中间值
    :return: 中间值的汉明重量
    '''
    m = 0
    while intermediate_value != 0:
        # print(type(intermediate_value))
        m=m+(intermediate_value & 1)
        intermediate_value >>= 1
    return m
def HW(data):
    '''

    :param data: 一组中间值
    :return: 一组中间值的汉明重量
    '''
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' should be a numpy ndarray.")
    if data.ndim == 1:
        new_data = np.zeros(len(data), dtype=float)
        for i in range(len(data)):
            new_data[i] = _hamming_weight(int(data[i]))
        return new_data
    if data.ndim == 2:
        new_data = np.zeros((data.shape(0),data.shape[1]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_data[i][j] = _hamming_weight(int(data[i][j]))
        return new_data

#
def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = np.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor
def one_key_guess(data, traces):
    one_key_guess_matrix = []
    for i in range(data.shape[1]):
        n = correlation_func(data[:, i], traces)
        one_key_guess_matrix.append(n)
    m = np.max(one_key_guess_matrix)
    t = np.where(one_key_guess_matrix == m)
    return t[0]
def get_imediate(label,klength):
    k_array = range(2**klength)
    data = np.zeros((label.shape[0],len(k_array)))
    for i in range(label.shape[0]):
        for j in range(len(k_array)):
            data[i][j] = int(label[i][0])^j
    return data
def hammingwhight(x: int, y: int) -> int:
    z = bin(x ^ y)[2:]
    count = 0
    for i in range(len(z)):
        if z[i] == '1':
            count += 1
    return count

def intermediate(pt, keyguess):
    '''

    :param pt: plaintext
    :param keyguess: key
    :return: intermediate value
    '''
    SBOX = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16],
        dtype=np.uint8
    )
    data = SBOX[pt ^ keyguess]
    return data


def getHW(byte):
    '''

    :param byte: input intermediate value
    :return: hw
    '''
    HW_table = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
                4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
                4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
                3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
                4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
                5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
                3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
                3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
                4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
                6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
                5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
                7, 7, 8]
    return HW_table[byte]

if __name__ == '__main__':
    # for i in range(len(pt)):

    pass

# def get_intermediate_value():
