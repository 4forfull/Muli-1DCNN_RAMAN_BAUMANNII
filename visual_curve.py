from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np


def vsraw(file_path):
    df = read_csv(file_path, header=None)
    df1 = df.T
    df1.columns = ['w', 'i']
    w = df1.iloc[:, :-1]  # Raman Shift
    i = df1.iloc[:, -1]  # Intensity

    vector = [rows.i for index, rows in df1.iterrows()]

    import scipy.signal as ss

    indices1 = ss.find_peaks_cwt(vector, np.arange(1, 30))
    indices = np.array(indices1) - 1

    value2 = []
    for a in indices1:
        value = df1.w.loc[a]
        value = int(value)
        value2.append(value)

    peak_intensity = []
    peak_tuple = []

    for index in indices:
        peak_intensity.append(i[index])
        peak_tuple.append((index, i[index]))

    for index, val in peak_tuple:
        if all(peak_intensity == val):
            peak_intensity = index

    plt.figure(figsize=(11, 6), dpi=100)
    ax = plt.subplot(1, 1, 1)  # 子图初始化

    indices = indices.tolist()
    plt.plot(w, i.T, color='red', markevery=indices, marker='x', markerfacecolor='black',
             markeredgecolor='black')  # T: transpose
    for i in range(len(peak_intensity)):
        ax.text(value2[i] * 1.03, peak_intensity[i] * 1.02, value2[i], fontsize=10, color="black", family='serif',
                weight="light", verticalalignment='center', horizontalalignment='right', rotation=0)

    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xlabel('Raman Shift (cm-1)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Raman Intensity', fontdict={'family': 'Times New Roman', 'size': 18})
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  # 设置上部坐标轴的粗细
    plt.rcParams.update({'font.size': 15})
    # plt.legend(frameon=False, prop={"family": "Times New Roman"})
    plt.savefig('raw.png')
    plt.close()
    # plt.show()


if __name__ == "__main__":
    file_path = 'D:/拉曼光谱/鲍曼不动杆菌/鲍曼-predict.csv'
    vsraw(file_path)