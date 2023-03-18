
import seaborn as sns
import numpy as np
import torch 


def show_matrix(matrix, ax=None, labels=None, title=None):
    if isinstance(matrix, torch.Tensor):
        matrix = np.asarray(matrix)
    vmax = np.max(np.abs(matrix))

    cmap = sns.diverging_palette(260, 10, as_cmap=True)
    if labels == None:
        labels = range(matrix.shape[-1])
    if ax == None:
        ax = sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            yticklabels=labels,
            xticklabels=labels,
        )
    else:
        sns.heatmap(
            matrix,
            cmap=cmap,
            vmax=vmax,
            vmin=-vmax,
            square=True,
            ax=ax,
            yticklabels=labels,
            xticklabels=labels,
        )

    if labels != None:
        ax.set_xticklabels(labels, rotation=80, fontsize=5)
        ax.set_yticklabels(labels, rotation=10, fontsize=5)
    if title != None:
        ax.set_title(title)
    return ax
