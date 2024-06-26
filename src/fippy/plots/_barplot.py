import matplotlib.pyplot as plt
import numpy as np
import math
from fippy.plots._utils import hbar_text_position, coord_height_to_pixels
import seaborn as sns

textformat = '{:5.2f}'  # TODO(gcsk): remove this line


def fi_sns_hbarplot(ex, ax=None, figsize=None, alpha=0.05, facecolor='darkgray', errcolor='black'):
    """Seaborn based function that plots rfi results

    Args:
        ex: Explanation object
        ax: ax to plot on?
        figsize: figsize of the form (width, height)
    """
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    df = ex.cis(type='two-sided', alpha=alpha)
    df.sort_values('importance', ascending=True, inplace=True)
    asymmetric_error = [(df['importance']-df['lower']).values, (df['upper']-df['importance']).values]
    df.reset_index(inplace=True)
    order = df.groupby('feature').mean().sort_values('importance', ascending=False).reset_index()['feature']
    with sns.axes_style('whitegrid'):
        plt.barh(y=df['feature'], width=df['importance'], xerr=asymmetric_error,
             color='black', facecolor=facecolor, capsize=0,
             linewidth=0, edgecolor='black', ecolor=errcolor)
        plt.title(f'{ex.ex_description}')
        # sns.barplot(ax=ax, x='importance', y='feature', data=df, order=order,
        #             yerr='width', color='black',
        #             facecolor='darkgray', capsize=0,
        #             linewidth=0, edgecolor='black',
        #             err_kws={'linewidth': 3, 'color': 'black'})
        sns.despine(left=True, bottom=True, ax=ax)
        ax.set(ylabel="Features", xlabel=f'Importance with {1-alpha} CIs')
        return ax

# err_kws={'linewidth': 3, 'color': 'black'}

### DEPRECATED

def fi_sns_gbarplot(dex, ax=None, figsize=None):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    df = dex.fi_decomp()
    df.reset_index(inplace=True)
    # df.sort_values('importance', axis=0, ascending=False, inplace=True)
    sns.barplot(x='importance', hue='component', y='feature',
                ax=ax, ci='sd', data=df,
                palette=sns.color_palette())
    return ax


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

def fi_sns_wbarplots(dex, fs=None, ax=None, figsize=None, col_wrap=5):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    df = dex.fi_decomp().reset_index()
    df.feature = df.feature.cat.set_categories(new_categories=df.component.cat.categories)
    for ix in ['total', 'remainder']:
        index = (df['component'] == ix)
        vals = df.loc[index, ['feature', 'component']].values
        df.loc[index, ['component', 'feature']] = vals

    if fs is not None:
        df = df[df['feature'].isin(fs)].copy()
        df.feature = df.feature.cat.remove_unused_categories().copy()
    g = sns.FacetGrid(df, col='feature', col_wrap=col_wrap)
    g.map(sns.barplot, 'importance', 'component',
          order=sorted(df.component.unique()))
