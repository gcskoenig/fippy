import matplotlib.pyplot as plt
import rfi.plots._snsstyle  # set default style
import numpy as np
import math
from rfi.plots._utils import hbar_text_position, coord_height_to_pixels, get_line_hlength
import seaborn as sns

textformat = '{:5.2f}'  # TODO(gcsk): remove this line


def fi_hbarplot(ex, textformat='{:5.2f}', ax=None, figsize=None):
    """
    Function that plots the result of an RFI computation as a barplot
    Args:
        figsize:
        ax:
        textformat:
        ex: Explanation object
    """

    df = ex.fi_means_stds()
    names = ex.fsoi_names
    rfis = df['mean'].to_numpy()
    stds = df['std'].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ixs = np.arange(rfis.shape[0] + 0.5, 0.5, -1)

    ax.barh(ixs, rfis, tick_label=names, xerr=stds, capsize=5)

    for jj in range(len(ax.patches)):
        rect = ax.patches[jj]
        tx, ty_lower = hbar_text_position(rect, y_pos=0.25)
        tx, ty_upper = hbar_text_position(rect, y_pos=0.75)
        pix_height = coord_height_to_pixels(ax, rect.get_height()) / 4
        pix_height = math.floor(pix_height)
        ax.text(tx, ty_upper, textformat.format(rect.get_width()),
                va='center', ha='center', size=pix_height)
        ax.text(tx, ty_lower, '+-' + textformat.format(stds[jj]),
                va='center', ha='center', size=pix_height)
    return ax


def fi_sns_hbarplot(ex, ax=None, figsize=None):
    """Seaborn based function that plots rfi results

    Args:
        ex: Explanation object
        ax: ax to plot on?
        figsize: figsize of the form (width, height)
    """
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    df = ex.fi_vals(return_np=False)
    df.reset_index(inplace=True)
    df.sort_values('importance', axis=0, ascending=False, inplace=True)
    sns.barplot(x='importance', y='feature', data=df, ax=ax, ci='sd')
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set(ylabel="", xlabel='Importance score {ex.ex_name}')
    return ax


def fi_sns_gbarplot(dex, ax=None, figsize=None):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    df = dex.fi_decomp()
    df.reset_index(inplace=True)
    # df.sort_values('importance', axis=0, ascending=False, inplace=True)
    sns.barplot(x='importance', hue='component', y='feature',
                ax=ax, ci='sd', data=df)
    return ax


# def container_hbarplot(exs, textformat='{:5.2f}'):
#     """Function that plots the results of multiple FI
#     computations (on the same features of interest).

#     Args:
#         exs: Iterable of explanations
#     """

#     fig, ax = plt.subplots(figsize=(16, 10))

#     ind = np.arange(len(exs[0].fsoi), 0, -1) - 1
#     height = ((1 / len(exs)) * 0.95)

#     ax.set_yticks(ind)
#     ax.set_yticklabels(exs[0].fsoi)
#     ax.legend()
#     fig.tight_layout()

#     containers = []
#     for jj in range(len(exs)):
#         barcontainer = ax.barh(ind + (jj * height), exs[jj].fi_means(), xerr=exs[jj].fi_stds(),
#                                height=height, label=exs[jj].ex_name, align='edge')
#         containers.append(barcontainer)

#     pix_height = math.floor(coord_height_to_pixels(ax, height) / 4)
#     # TODO(gcsk): set font size to pix_height

#     # TODO(gcsk)
#     # barcontainers have patches and errorbars
#     # modify code above to automatically label the bars
#     # docu: https://matplotlib.org/3.1.1/api/container_api.html#matplotlib.container.BarContainer
#     for jj in range(len(exs)):
#         patches = containers[jj].patches
#         # errbar_lines = containers[jj].errorbar.lines[2].segments
#         for kk in range(len(patches)):
#             # errbar_length = get_line_hlength(errbar_lines[kk])
#             tx, ty_lower = hbar_text_position(patches[kk], y_pos=0.25)
#             tx, ty_upper = hbar_text_position(patches[kk], y_pos=0.75)
#             ax.text(0, ty_upper, textformat.format(patches[kk].get_width()),
#                     va='center', ha='left', size=pix_height)
#             # ax.text(tx, ty_lower, '+-'+ textformat.format(errbar_length), va='center', ha='center', size=pix_height)

#     plt.show()