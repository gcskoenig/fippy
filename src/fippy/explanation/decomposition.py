from fippy.explanation.explanation import Explanation
import pandas as pd
from fippy.plots._barplot import fi_sns_gbarplot, fi_sns_wbarplots
import numpy as np


class DecompositionExplanation(Explanation):
    """Multiple explanations can be stored in a Explanation
    Container

    Attributes:
        fsoi: Features of interest.
        scores: losses on perturbed
            (nr_fsoi, nr_components, nr_permutations, nr_runs)
        ex_name: Explanation description
    """
    def __init__(self, fsoi, scores, ex_name=None):
        Explanation.__init__(self, fsoi, scores, ex_name=ex_name)

    @staticmethod
    def from_csv(path, ex_name=None):
        index_candidates = ['component', 'ordering', 'sample', 'i']
        scores = pd.read_csv(path)
        index_names = list(index_candidates[np.isin(index_candidates, scores.columns)])
        scores = scores.set_index(index_names['component', 'ordering', 'sample'])
        ex = DecompositionExplanation(scores.columns, scores, ex_name=ex_name)
        return ex

    # def _check_shape(self):
    #     if len(self.lss.shape) != 4:
    #         raise RuntimeError('.lss has shape {self.lss.shape}.'
    #                            'Expected 4-dim.')

    # def _create_index(self, dims=[0, 1, 3]):
    #     names = ['feature', 'component', 'permutation', 'run']
    #     runs = range(self.lss.shape[3])
    #     permutations = range(self.lss.shape[2])
    #     lists = [self.fsoi_names, self.component_names, permutations, runs]

    #     names_sub = [names[i] for i in dims]
    #     lists_sub = [lists[i] for i in dims]
    #     index = utils.create_multiindex(names_sub, lists_sub)
    #     return index

    def fi_vals(self, fnames_as_columns=True):
        if fnames_as_columns:
            df = self.scores.copy()
            df = df.groupby(['component', 'sample']).mean()
            dfc = df.reset_index()
            dfc = dfc[dfc['component'] == 'total']
            dfc = dfc.set_index(['sample'], drop=True)
            dfc = dfc.drop(columns=['component'])
            return dfc
        else:
            df = self.fi_decomp()
            dfc = df.reset_index()
            dfc = dfc[dfc['component'] == 'total']
            dfc = dfc.set_index(['feature', 'sample'], drop=True)
            dfc = dfc.drop(columns=['component'])
            return dfc

    def fi_means_stds(self):
        pass

    def fi_cols_to_index(self):
        df = self.scores.copy()
        sr = pd.concat([df[col] for col in df],
                       keys=df.columns,
                       names=['feature'])
        df = pd.DataFrame(sr, columns=['importance'])
        return df.copy()

    def fi_decomp(self, sorted=True):
        df = self.fi_cols_to_index()
        df = df.groupby(['feature', 'component', 'sample']).mean()
        if not sorted:
            return df
        else:
            # sort by total importance
            fi_ordering = df.loc[:, 'total', :].groupby(level=0).mean()
            fi_ordering = fi_ordering.sort_values('importance',
                                                  ascending=False)
            fi_ordering = list(fi_ordering.index.values)

            cmpns = set(df.index.get_level_values('component'))
            summaries = list(set(['total', 'remainder']).intersection(cmpns))
            cmpn_ordering = summaries + fi_ordering
            remainder = list(set(df.index.levels[1]).difference(cmpn_ordering))
            cmpn_ordering = cmpn_ordering + remainder

            fs_indx = df.index.levels[0].astype('category')
            fs_indx = fs_indx.reorder_categories(fi_ordering)
            cmpn_indx = df.index.levels[1].astype('category')
            cmpn_indx = cmpn_indx.reorder_categories(cmpn_ordering)
            df.index = df.index.set_levels([fs_indx,
                                            cmpn_indx,
                                            df.index.levels[2]])
            df = df.sort_index()
            return df

    def decomp_hbarplot(self, figsize=None, ax=None):
        """
        Advanced hbarplot for multiple RFI computations

        Args:

        """
        return fi_sns_gbarplot(self, figsize=figsize, ax=ax)

    def decomp_wbarplots(self, figsize=None, ax=None, fs=None, col_wrap=5):
        """
        multiplot where the features that shall be visualized can be specified
        using the fs argument.
        fs include "total" or "remainder" if the respective components
        are part of the decomposition
        """
        return fi_sns_wbarplots(self, figsize=figsize, ax=ax, fs=fs, col_wrap=col_wrap)

    def fi_means_quantiles(self):
        """
        not implemented yet
        """
        raise NotImplementedError('fi_means_quantiles is not implemented for DecompositionExplanations yet.')
