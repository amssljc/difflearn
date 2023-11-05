
import seaborn as sns
import numpy as np
import torch 
import matplotlib.pyplot as plt

def show_matrix(matrix, ax=None, labels='number', label_lenth = 11,  title=None):
    if isinstance(matrix, torch.Tensor):
        matrix = np.asarray(matrix)
    vmax = np.max(np.abs(matrix))
    p  = matrix.shape[-1]
    cmap = sns.diverging_palette(260, 10, as_cmap=True)
    if labels == 'number':
        labels  = list(np.linspace(0,p,label_lenth,dtype=int))
    if ax == None:
        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
        )

        
    if labels is not False:
        xtick_positions = list(np.linspace(0,p,label_lenth,dtype=int))
        ytick_positions = list(np.linspace(0,p,label_lenth,dtype=int))

        ax.set_xticks(xtick_positions)
        ax.set_yticks(ytick_positions)
        ax.set_xticklabels(labels, rotation=80, fontsize=5)
        ax.set_yticklabels(labels, rotation=10, fontsize=5)
    if title != None:
        ax.set_title(title)
    return ax
