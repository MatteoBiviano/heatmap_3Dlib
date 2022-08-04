import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl

def usage():
    """
    Print the usage of heatmap_3d function
    """
    print("-- heatmap_3d function must used in this way:")
    print("   heatmap_3d(path: str, optimal: list, param1: str, param2: str, param3: str, modul: int, metric: str,  crt : (str, str) = (), color_map: str = PuBu, define_opt: list = [])")
    print("-- explanation about parameters:")
    print(" :param path: path for the dataframe contains grid search results \
              dataframe must be in one of these form: \
              - <name_metric_tuned, criterion, param1, param2, param3> \
              - <param1, param2, param3> \
            :param optimal: list of optimal values combination (e.g. [max_depth, min_samples_split, min_samples_leaf])\
            :param param1: first parameter tuned (x-axis)\
            :param param2: second parameter tuned (y-axis)\
            :param param3: third parameter tuned (z-axis)\
            :param modul: number of spaces between each pair of heatmaps plotted\
            :param metric: metric used in grid search (e.g. f1, accuracy, ...)\
            :param crt: (optional) pair (name_column, criterion) where name is the name used for identify the column in dataframe, while criterion is the criterion used in grid search (e.g. gini, entropy, ...)\
            :param color_map: (optional) color template for the heatmap\
            :param define_opt: (optional) is the list of RGBA using for identify optimal value (e.g. [1, 0, 0, 1]). If is not defined, alpha=1 identify optimal value"
    )

def trunc_val(n, dec=0):
    """
    Truncate the number of specific decimal

    :param n: number to be truncate
    :param dec: decimals to use for the truncation
    :return: the truncated value
    """

    if isinstance(dec, int):
        if dec ==0:
            return math.trunc(n)
        elif dec >0:
            f = 10.0 ** dec
            return math.trunc(n * f) / f
    raise ValueError("value not correct")

def heatmap_3d(path: str, optimal: list, param1: str, 
                param2: str, param3: str, modul: int, metric: str,  
                crt : tuple = (), color_map: str = "PuBu", define_opt: list = []):
    """
    Plot 3d heatmap, for visualize grid search results.

    :param path: path for the dataframe contains grid search results
    dataframe must be in one of these form:
        - <name_metric_tuned, "criterion", param1, param2, param3>
        - <param1, param2, param3>
    :param optimal: list of optimal values combination (e.g. [max_depth, min_samples_split, min_samples_leaf])
    :param param1: first parameter tuned (x-axis)
    :param param2: second parameter tuned (y-axis)
    :param param3: third parameter tuned (z-axis)
    :param modul: number of spaces between each pair of heatmaps plotted
    :param metric: metric used in grid search (e.g. "f1", "accuracy", ...)
    :param crt: (optional) pair (name_column, criterion) where name is the name used for identify the column in dataframe, while criterion is the criterion used in grid search (e.g. "gini", "entropy", ...)
    :param color_map: (optional) color template for the heatmap
    :param define_opt: (optional) is the list of RGBA using for identify optimal value (e.g. [1, 0, 0, 1]). If is not defined, alpha=1 identify optimal value
    :return: tuple <axis, figure, colorbar>
    """


    
    # Read dataframe
    df = pd.read_csv(path)
    if crt :
        df_ = df[df[crt[0]] == crt[1]].copy()
    else:
        df_ = df
    # Define dimensions
    par1 = list(df_[param1])
    dim_x = len(set(par1))
    dim_x = dim_x + (modul * (dim_x - 1))

    # Define modul = n° space + 1)
    modul = modul +1

    par2= list(df_[param2])
    dim_y = len(set(par2))

    par3= list(df_[param3])
    dim_z = len(set(par3))

    _score = list(df_[metric])
    distinct_ = sorted(list(set(_score)))
    dim_score = len(distinct_)
    
    # Define colormap
    viridisBig = cm.get_cmap(color_map, 521)
    newcmp = ListedColormap(viridisBig(np.linspace(0, 1, dim_score)))
    list_colors = newcmp.colors

    # Add alpha channel in color_map
    dc_ = {}
    idx = 0
    for i in distinct_:
        te = list(list_colors[idx])
        te[3]=0.7
        dc_[trunc_val(i, 4)] = te
        idx = idx + 1

    par1 = sorted(list(set(par1)))
    par2 = sorted(list(set(par2)))
    par3 = sorted(list(set(par3)))
    
    # Color grid
    colors = np.empty([dim_x, dim_y, dim_z, 4], dtype=np.float32)
    
    # Make combination of three parameters
    real_i=0
    for i in range(0, dim_x):
        if i%modul==0:
            for j in range(0, dim_y):
                    for k in range(0, len(colors[i, j])):
                        colors[i, j][k] = [par1[real_i],par2[j],par3[k], 0]
            real_i = real_i + 1
    
    
    """
    ---- Optional
    for i in range(0, dim_x):
        if i%modul!=0:
            for j in range(0, dim_y):
                for k in range(0, len(colors[i, j])):
                        colors[i, j][k] = [None, None, None, 0]
    """                    
    for i in range(0, dim_x, modul):
            for j in range(0, dim_y):
                    for k in range(0, len(colors[i, j])):
                        value = df_[(df_[param1]==colors[i,j][k][0]) & (df_[param2]==colors[i,j][k][1]) & (df_[param3]==colors[i,j][k][2])][metric]
                        truncated_val = trunc_val(value.values[0], 4)
                        prova = dc_[truncated_val]
                        if(colors[i,j][k][0] == optimal[0] and colors[i,j][k][1] == optimal[1] and colors[i,j][k][2] == optimal[2]):
                            
                            if len(define_opt)==0:
                                # Optimal values defined by alpha = 1 
                                prova[3]=1
                            else:
                                prova = define_opt
                            colors[i, j][k] = np.array(prova)
                            
                        else:
                            prova[3]=0.7
                            colors[i, j][k] = np.array(prova)
    
    fils = np.ones([dim_x, dim_y, dim_z], dtype=np.bool)
    for i in range(0, dim_x):
        if i%modul!=0:
            for j in range(0, dim_y):
                for k in range(0, len(colors[i, j])):
                       fils[i][j][k] = 0
    fig = plt.figure(figsize = (10, 10))

    ax = fig.add_subplot('111', projection='3d')
    ax.voxels(fils, facecolors=colors, edgecolors='k')

    norm = matplotlib.colors.Normalize(vmin=min(_score), vmax=max(_score))
    m = mpl.cm.ScalarMappable(cmap=newcmp, norm=norm)
    m.set_array([])
    cbr = plt.colorbar(m, fraction=0.03, pad=0.04)
    cbr.ax.set_title(metric.upper())
    return ax, fig, cbr


def heatmap_bi(dataset: pd.DataFrame, optimal: list, 
                param1: str, param2: str, param3: str, 
                modul: int, metric: str,  crt :  tuple = (), 
                color_map: str = "PuBu", define_opt: list = []):
    """
    Plot 3d heatmap, for visualize grid search results in PowerBi.

    :param dataset: dataframe contains grid search results
    :param optimal: list of optimal values combination (e.g. [max_depth, min_samples_split, min_samples_leaf])
    :param param1: first parameter tuned (x-axis)
    :param param2: second parameter tuned (y-axis)
    :param param3: third parameter tuned (z-axis)
    :param modul: number of spaces between each pair of heatmaps plotted
    :param metric: metric used in grid search (e.g. "f1", "accuracy", ...)
    :param crt: (optional) pair (name_column, criterion) where name is the name used for identify the column in dataframe, while criterion is the criterion used in grid search (e.g. "gini", "entropy", ...)
    :param color_map: (optional) color template for the heatmap
    :param define_opt: (optional) is the list of RGBA using for identify optimal value (e.g. [1, 0, 0, 1]). If is not defined, alpha=1 identify optimal value
    :return: tuple <axis, figure, colorbar>
    """
  
    # Read dataframe
    df = dataset
    if crt:
        df_ = df[df[crt[0]] == crt[1]].copy()
    else:
        df_ = df
    # Define dimensions
    par1 = list(df_[param1])
    dim_x = len(set(par1))
    dim_x = dim_x + (modul * (dim_x - 1))

    # Define modul = n° space + 1)
    modul = modul +1

    par2= list(df_[param2])
    dim_y = len(set(par2))

    par3= list(df_[param3])
    dim_z = len(set(par3))

    _score = list(df_[metric])
    distinct_ = sorted(list(set(_score)))
    dim_score = len(distinct_)

    # Define colormap
    viridisBig = cm.get_cmap(color_map, 521)
    newcmp = ListedColormap(viridisBig(np.linspace(0, 1, dim_score)))
    list_colors = newcmp.colors

    # Add alpha channel in color_map
    dc_ = {}
    idx = 0
    for i in distinct_:
        te = list(list_colors[idx])
        te[3]=0.7
        dc_[trunc_val(i, 4)] = te
        idx = idx + 1

    par1 = sorted(list(set(par1)))
    par2 = sorted(list(set(par2)))
    par3 = sorted(list(set(par3)))

    # Color grid
    colors = np.empty([dim_x, dim_y, dim_z, 4], dtype=np.float32)

    # Make combination of three parameters
    real_i=0
    for i in range(0, dim_x):
        if i%modul==0:
            for j in range(0, dim_y):
                    for k in range(0, len(colors[i, j])):
                        colors[i, j][k] = [par1[real_i],par2[j],par3[k], 0]
            real_i = real_i + 1

    for i in range(0, dim_x, modul):
            for j in range(0, dim_y):
                    for k in range(0, len(colors[i, j])):
                        value = df_[(df_[param1]==colors[i,j][k][0]) & (df_[param2]==colors[i,j][k][1]) & (df_[param3]==colors[i,j][k][2])][metric]
                        truncated_val = trunc_val(value.values[0], 4)
                        prova = dc_[truncated_val]
                        if(colors[i,j][k][0] == optimal[0] and colors[i,j][k][1] == optimal[1] and colors[i,j][k][2] == optimal[2]):
                            
                            if len(define_opt)==0:
                                # Optimal values defined by alpha = 1 
                                prova[3]=1
                            else:
                                prova = define_opt
                            colors[i, j][k] = np.array(prova)
                            
                        else:
                            prova[3]=0.7
                            colors[i, j][k] = np.array(prova)

    fils = np.ones([dim_x, dim_y, dim_z], dtype=bool)
    for i in range(0, dim_x):
        if i%modul!=0:
            for j in range(0, dim_y):
                for k in range(0, len(colors[i, j])):
                        fils[i][j][k] = 0
    fig = plt.figure(figsize = (10, 10))

    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(fils, facecolors=colors, edgecolors='k')

    norm = matplotlib.colors.Normalize(vmin=min(_score), vmax=max(_score))
    m = mpl.cm.ScalarMappable(cmap=newcmp, norm=norm)
    m.set_array([])
    cbr = plt.colorbar(m, fraction=0.03, pad=0.1)
    cbr.ax.set_title(metric.upper())
    return ax, fig, cbr