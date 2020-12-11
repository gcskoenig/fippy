import matplotlib.pyplot as plt
import rfi.plots._snsstyle #set default style
import numpy as np
import math
from rfi.plots._utils import hbar_text_position, coord_height_to_pixels, get_line_hlength
textformat='{:5.2f}' # TODO(gcsk): remove this line

def rfi_hbarplot(ex, textformat='{:5.2f}'):
    '''Function that plots the result of an RFI computation
    as a barplot

    Args:
        ex: Explanation object
    '''

    rfis = ex.rfi_means()
    stds = ex.rfi_stds()
    names = ex.rfi_names()
    
    fig, ax = plt.subplots()

    ixs = np.arange(rfis.shape[0] + 0.5, 0.5, -1)

    ax.barh(ixs, rfis, tick_label=names, xerr=stds, capsize=5)

    for jj in range(len(ax.patches)):
        rect = ax.patches[jj]
        tx, ty_lower = hbar_text_position(rect, y_pos=0.25)
        tx, ty_upper = hbar_text_position(rect, y_pos=0.75)
        pix_height = math.floor(coord_height_to_pixels(ax, rect.get_height())/4)
        ax.text(tx, ty_upper, textformat.format(rect.get_width()), 
                va='center', ha='center', size=pix_height)
        ax.text(tx, ty_lower, '+-'+ textformat.format(stds[jj]), 
                va='center', ha='center', size=pix_height)
    return ax


def container_hbarplot(exs, textformat='{:5.2f}'):
    """Function that plots the results of multiple RFI
    computations (on the same features of interest).

    Args:
        exs: Iterable of explanations
    """

    fig, ax = plt.subplots(figsize=(16,10))

    ind = np.arange(len(exs[0].fsoi), 0, -1) - 1
    height = ((1 / len(exs)) * 0.95)

    ax.set_yticks(ind)
    ax.set_yticklabels(exs[0].fsoi)
    ax.legend()
    fig.tight_layout()

    containers = [] 
    for jj in range(len(exs)):
        barcontainer = ax.barh(ind + (jj*height), exs[jj].rfi_means(), xerr=exs[jj].rfi_stds(),
                               height=height, label=exs[jj].ex_name, align='edge')
        containers.append(barcontainer)

    pix_height = math.floor(coord_height_to_pixels(ax, height)/4)
    # TODO(gcsk): set font size to pix_height

    # TODO(gcsk)
    # barcontainers have patches and errorbars
    # modify code above to automatically label the bars 
    # docu: https://matplotlib.org/3.1.1/api/container_api.html#matplotlib.container.BarContainer
    for jj in range(len(exs)):
        patches = containers[jj].patches
        #errbar_lines = containers[jj].errorbar.lines[2].segments
        for kk in range(len(patches)):
            #errbar_length = get_line_hlength(errbar_lines[kk])
            tx, ty_lower = hbar_text_position(patches[kk], y_pos=0.25)
            tx, ty_upper = hbar_text_position(patches[kk], y_pos=0.75)
            ax.text(0, ty_upper, textformat.format(patches[kk].get_width()), 
                    va='center', ha='left', size=pix_height)
            #ax.text(tx, ty_lower, '+-'+ textformat.format(errbar_length), 
            #        va='center', ha='center', size=pix_height)  

    plt.show()

# def rfi_barplot(rfis, fnames, rfinames, savepath, figsize=(16,10), textformat='{:5.2f}')
#     """
#     rfis: list of tuples (means, stds)

#     """

#     ind = np.arange(len(fnames))  # the x locations for the groups
#     width = (1/len(rfinames)*0.95)  # the width of the bars

#     fig, ax = plt.subplots()
#     rects = []
#     for rfi_ind in np.arange(0, len(rfinames), 1):
#         print(rfi_ind)
#         rects_inx = ax.bar(ind + width*(rfi_ind+0.5), rfis[rfi_ind][0], width, 
#           yerr=rfis[rfi_ind][1], label=rfinames[rfi_ind])
#         rects.append(rects_inx)

#     ax.set_ylabel('Importance')
#     ax.set_title('RFIs Plot')
#     ax.set_xticks(ind)
#     ax.set_xticklabels(fnames)
#     ax.legend()


#     def autolabel(rects, xpos=0):
#         """
#         Attach a text label above each bar in *rects*, displaying its height.
#         """
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate(textformat.format(height),
#                         xy=(rect.get_x(), height),
#                         xytext=(3, 4),  # use 3 points offset 
#                         #previously in xpos of xytext: +rect.get_width()/2
#                         textcoords="offset points",  # in both directions
#                         va='bottom')


#     for ii in range(len(rects)):
#         autolabel(rects[ii], xpos=ii)

#     fig.tight_layout()
#     plt.savefig(savepath, figsize=figsize)
#     plt.show()