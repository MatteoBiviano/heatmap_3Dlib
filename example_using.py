

import heatmap3Dlib
from heatmap3Dlib import plot3D as p3D

x_ticks = ["", "None", "       2", "","5", "       10", "", "15 ", "       20"]
y_ticks = ["", "", "2", "5",  "10", "15", "20"]
z_ticks = ["    1","     5", "     10", "    15", "   20"]

import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("Examples/recall_resultDT.csv")
crt = ("criterion",'gini')
ax, fig, cbr = p3D.heatmap_bi(dataset = dataset, metric = "recall", optimal = [0, 2, 20], 
            crt = crt, 
            param1 = "max_depth",
            param2 = "min_samples_split",
            param3 = "min_samples_leaf",
            modul=2)
ax.set_xticklabels(x_ticks, fontsize=12)
ax.set_yticklabels(y_ticks, fontsize=12)
ax.set_zticklabels(z_ticks, fontsize=12)
ax.set_xlabel("max_depth", fontsize=15, labelpad=10)
ax.set_ylabel("min_samples_split", fontsize=15, labelpad=10)
ax.set_zlabel("min_samples_leaf", fontsize=15, labelpad=10)
ax.set_title(f"Criterion - {crt[1]}", fontsize=15, loc='center', pad=15)
ax.view_init(30,300)
plt.show()

