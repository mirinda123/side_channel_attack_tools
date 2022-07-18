#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sca 
@File    ：trace_cut.py
@Author  ：suyang
@Date    ：2022/3/24 10:36 
'''
import numpy as np
from tqdm import tqdm, trange
def get_trace_segment(n,url_trace,leftmin,rightmax):
    list = []
    for j in trange(n - 1):
        sub_list = []
        arr = np.load(url_trace + r"passarrPart{0}.npy".format(j))
        # for i in range(arr.shape[0])
        cut_arr = arr[:,leftmin:rightmax]
        # Save an array to a binary file in NumPy.npy format.
        np.save(url_trace+"cut_arr{0}.npy".format(j), cut_arr)
    return arr[leftmin:rightmax]
