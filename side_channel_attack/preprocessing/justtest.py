from  pre_traces import to_one2
import numpy as np
import matplotlib.pyplot as plt

tracepass1 = np.load(
    r"D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\tutorials\aes_random_PICO5000_125MHZarrPart0.npy")
plt.plot(tracepass1[0])

tracepass2 = to_one2(tracepass1)
plt.plot(tracepass2[0])

plt.show()



