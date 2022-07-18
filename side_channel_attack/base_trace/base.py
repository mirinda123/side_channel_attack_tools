import numpy as np
import estraces as traces
# 这块没有用，因为filetype包暂未支持npy文件
# import filetype
import logging
import os
import re
import sys
from datetime import datetime
import zipfile
# ths = scared.traces.read_ths_from_ets_file()
from numpy import test
# import pandas as pd
import estraces as traces


# so i need a tool to make others into npy

def ensure_extension(path):

    root, ext = os.path.splitext(path)
    if ext == '.npy':
        return path
    elif ext =='.ets':
        return traces.read_ths_from_ets_file(path).samples[:]
    elif ext == '.csv':
        filepath = 'G:/py+idea/python/sidechannel/basetrace/exsample.csv'
        t1 = np.genfromtxt(filepath, delimiter=',', encoding='utf-8', skip_header=2)
        return t1[:,2]
        # print(t1)
    # if ext != '.npy':
    #     raise IOError("  not exist or is not a path")
    else:
        raise IOError("  not exist or is not a path")




def is_bytes_array(traces):
    if not isinstance(traces, np.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(traces)}.')
    if not traces.dtype == 'uint8':
        raise ValueError(f'array should be a byte array with uint8 dtype, not {traces.dtype}.')
    return True


def delete_one_trace(traces,order):
    # newtraces = []
    newtraces = np.delete(traces,order,axis=0)
    return newtraces

def delete_onemore_trace(traces,order1,order2):
    # newtraces = []
    newtraces = np.delete(traces,[order1,order2],axis=0)
    return newtraces



def get_traces_num(traces):
    # with open('gpu_sca.npy', "rb") as f:
    #     kind = filetype.guess(f)
    #     if kind is None:
    #         print('Cannot identyfy the file type')
    #         return
    # print(kind.extension)
    # print(kind.mime)
    # print('the correct extension')
    # trace_name = project.ensure_cwp_extension(filename)
    return np.shape(traces)[0]

def get_trace_sample(trace):
    return np.size(trace)

def get_traces_sample(traces):
    return np.shape(traces)[1]


def get_traces_mean(traces):
    return np.mean(traces, axis=0)


def get_traces_abs(traces):
    return np.abs(traces)

