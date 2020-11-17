import matplotlib.pyplot as plt
import rfi.plots._snsstyle #set default style
import numpy as np

def rfi_hbarplot(ex):
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

	for jj in range(len(ixs)):
		rect = ax.barh(ixs[jj], rfis[jj], tick_label=names[jj], xerr=stds[jj], capsize=5)

	# ## determine height of bar in pixels
	# def autolabel(rects, xpos=0):
 #        """
 #        Attach a text label above each bar in *rects*, displaying its height.
 #        """
 #        for rect in rects:
 #            height = rect.get_height()
 #            ax.annotate(textformat.format(height),
 #                        xy=(rect.get_x(), height),
 #                        xytext=(3, 4),  # use 3 points offset 
 #                        #previously in xpos of xytext: +rect.get_width()/2
 #                        textcoords="offset points",  # in both directions
 #                        va='bottom')


	for jj in range(len(ixs)):
		ax.text(rfis[jj], ixs[jj] + 0.1, 'mean: {}'.format(rfis[jj]), va='center')
		ax.text(rfis[jj], ixs[jj] - 0.1, 'std: {}'.format(stds[jj]), va='center')

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
#         	yerr=rfis[rfi_ind][1], label=rfinames[rfi_ind])
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