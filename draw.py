import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
figpath = '/data/users/gaoli/exp_Robust/figures/test.png'

x = [i for i in range(6)]
id2method = {
    "0": "Ours",
    "1": "FS",
    "2": "Co+FS",
    "3": "DT+FS",
    "4": "ODDS+FS",
    "5": "SMOTE+FS",
    "6": "DT+ODDS+FS",
    "7": "ETA",
    "8": "DT+ODDS+ETA"
}

color_dict= {
    "0": "#c82423", #
    "1": "#0088C4", #
    "2": "#00A7D0",
    "3": "#00C3C4",
    "4": "#57784B",  
    "5": "#8AAD7C", 
    "6": "#5DA37E", 
    "7": "#5E4783", #
    "8": "#A5538B" #
}

linestyle_dict= {
    "0": "--", 
    "1": "--", 
    "2": "--",
    "3": "--",
    "4": "--",  
    "5": "--", 
    "6": "--", 
    "7": "--", 
    "8": "--" 
}

marker_dict= {
    "0": "o", # 1
    "1": "^", # 2
    "2": "^", # 2
    "3": "^", # 2
    "4": "*",  # 2
    "5": "*", # 2
    "6": "*", # 2
    "7": "s", # 3
    "8": "s" # 3
}

markersize_all = 5

linewidth_all = 2.5

alpha_dict= {
    "0": 0.8, #
    "1": 0.6, #
    "2": 0.6,
    "3": 0.6,
    "4": 0.6,  #
    "5": 0.6, 
    "6": 0.6, 
    "7": 0.6, #
    "8": 0.6 
}

result = {
    "f1_DoHBrw": {
        "0": [0.779   , 0.365  , 0.471  , 0.394  , 0.224  , 0.668  , 0.486  , 0.705  , 0.705],
        "1": [0.781   , 0.392  , 0.380  , 0.477  , 0.103  , 0.666  , 0.542  , 0.704  , 0.704],
        "2": [0.779   , 0.331  , 0.429  , 0.468  , 0.177  , 0.667  , 0.496  , 0.701  , 0.703],
        "3": [0.782   , 0.356  , 0.446  , 0.445  , 0.062  , 0.666  , 0.458  , 0.693  , 0.704],
        "4": [0.776   , 0.360  , 0.487  , 0.353  , 0.066  , 0.663  , 0.359  , 0.652  , 0.612],
        "5": [0.770   , 0.338  , 0.494  , 0.341  , 0.041  , 0.659  , 0.392  , 0.708  , 0.422]
    },
    "f1_AndMal": {
        "0": [0.763  , 0.179  , 0.400  , 0.374  , 0.035  , 0.065  , 0.525  , 0.482  , 0.400],
        "1": [0.720  , 0.142  , 0.318  , 0.292  , 0.008  , 0.065  , 0.310  , 0.377  , 0.416],
        "2": [0.705  , 0.119  , 0.200  , 0.243  , 0.009  , 0.064  , 0.349  , 0.392  , 0.361],
        "3": [0.703  , 0.087  , 0.087  , 0.206  , 0.011  , 0.063  , 0.337  , 0.236  , 0.259],
        "4": [0.726  , 0.080  , 0.079  , 0.148  , 0.007  , 0.063  , 0.060  , 0.163  , 0.160],
        "5": [0.748  , 0.063  , 0.056  , 0.099  , 0.005  , 0.063  , 0.052  , 0.122  , 0.189]
    }
}


def get_f1_score(method_id, dataset):
    y = []
    for idx in result[dataset]:
        y.append(result[dataset][idx][method_id])
    return y

fig = plt.figure(figsize=(6.7, 2.3))
ax = plt.subplot(1, 2, 1)
for method_id in [0, 7, 8, 1, 2, 3, 4, 5, 6]:
    ax.plot(x, get_f1_score(method_id, "f1_DoHBrw"), 
        label=id2method[str(method_id)],
        color=color_dict[str(method_id)],
        linestyle=linestyle_dict[str(method_id)],
        marker=marker_dict[str(method_id)],
        alpha=alpha_dict[str(method_id)],
        markersize=markersize_all,
        linewidth=linewidth_all
    )

ax.set_xticks([1, 3, 5])# 坐标轴数值
ax.set_xticklabels(["25\%", "35\%", "45\%"])# 坐标轴数值显示
ax.set_yticks([0, 0.25, 0.5, 0.75])# 坐标抽数值

ax.legend(loc='center left', 
    bbox_to_anchor=(-0.25, 1.45),
    ncol=3, fontsize=14,
    frameon=True)
ax.set_xlabel("The noise ratio",fontsize=16)
ax.set_ylabel('F1-score',fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.spines["bottom"].set_linewidth(1.3)
ax.spines["left"].set_linewidth(1.3)
ax.spines["right"].set_linewidth(1.3)
ax.spines["top"].set_linewidth(1.3)
ax.set_title("DoHBrw dataset", fontsize=16)

ax = plt.subplot(1, 2, 2)
for method_id in [0, 7, 8, 1, 2, 3, 4, 5, 6]:
    ax.plot(x, get_f1_score(method_id, "f1_AndMal"), 
        label=id2method[str(method_id)],
        color=color_dict[str(method_id)],
        linestyle=linestyle_dict[str(method_id)],
        marker=marker_dict[str(method_id)],
        alpha=alpha_dict[str(method_id)],
        markersize=markersize_all,
        linewidth=linewidth_all
    )

ax.set_xticks([1, 3 ,5])
ax.set_xticklabels(["25\%", "35\%","45\%"])
ax.set_yticks([])
ax.set_xlabel("The noise ratio",fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.spines["bottom"].set_linewidth(1.3)
ax.spines["left"].set_linewidth(1.3)
ax.spines["right"].set_linewidth(1.3)
ax.spines["top"].set_linewidth(1.3)
ax.set_title("IDS+AndMal dataset", fontsize=16)

#ax.legend(title='Order')
# ax.set(xlabel='Voltage (mV)')
# ax.set(ylabel='Current ($\mu$A)')
# ax.autoscale(tight=True)
fig.savefig(figpath, dpi=300)
print("figure saved in ", figpath)