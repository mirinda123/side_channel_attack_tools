import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as transforms
if __name__ == '__main__':
    trace12 = np.load(r'D:/side_channel_attack/result/bestorder2trace12.npy')

    plt.rcParams['figure.figsize'] = (12.0, 7.0)
    f, ax = plt.subplots(1, 1)
    # 画出主图
    plt.plot(trace12[1:20000])
    plt.subplots_adjust(left=0.10, right=0.95, top=0.9, bottom=0.2)

    # 画出两条红线
    ax.axhline(y=4.5, ls='--', c='red',linewidth=2)
    ax.axhline(y=-4.5, ls='--', c='red', linewidth=2)

    # 设置整个图像
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, 4.5, "{:.1f}".format(4.5), fontsize=12,color="red", transform=trans,
            ha="right", va="center")
    ax.text(0, -4.5, "{:.1f}".format(-4.5), fontsize=12, color="red", transform=trans,
            ha="right", va="center")
    # 设定x轴的范围
    ax.set_xlim((0, 20000))
    ax.set_ylim((-5, 5))
    # ax.set_ylim((-250, 200))
    #
    # ax.set_ylim((0,5.5))
    x = [0,5000, 10000, 15000]
    # 为后边科学计数法做铺垫
    def formatnum(x, pos):
        return '$%.1f$x$10^{4}$' % (x / 10000)


    plt.xticks(x, ('0','$0.5$x$10^{4}$', '$1.0$x$10^{4}$', '$1.5$x$10^{4}$'))
    # formatter1 = FuncFormatter(formatnum)
    # ax.xaxis.set_major_formatter(formatter1)

    # plt.xlabel("(a)  50000  traces", labelpad=15, fontdict={'family': 'Times New Roman', 'size': 40})
    # fontweight = 'bold
    plt.ylabel("$t$-statistic", fontsize=13, labelpad=10)

    plt.savefig('./bestorder2trace12.png', dpi=120, bbox_inches='tight')
    # plt.title(\"TTest (A,B) result trace\")
    plt.show()
