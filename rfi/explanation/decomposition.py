from rfi.explanation.explanation import Explanation
import pandas as pd
import numpy as np
import itertools
from rfi.plots._barplot import fi_sns_gbarplot, fi_sns_hbarplot
import rfi.utils as utils


class DecompositionExplanation(Explanation):
    """Multiple explanations can be stored in a Explanation
    Container

    Attributes:
        fsoi: Features of interest.
        lss: losses on perturbed
            (nr_fsoi, nr_components, nr_permutations, nr_runs)
        ex_name: Explanation description
        fsoi_names: feature of interest names
        component_names: names of the decompotion components
    """
    def __init__(self, fsoi, scores, ex_name=None):
        Explanation.__init__(self, fsoi, scores, ex_name=ex_name)

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

    def fi_vals(self, return_np=False):
        if return_np:
            raise NotImplementedError('np.array support not implemented.')
        df = self.fi_decomp()
        dfc = df.reset_index()
        dfc = dfc[dfc['component'] == 'total']
        dfc = dfc.set_index(['feature', 'run'], drop=True)
        dfc = dfc.drop(columns=['component'])
        return dfc

    def fi_means_stds(self):
        pass

    def fi_decomp(self):
        return self.scores

    def decomp_hbarplot(self, figsize=None, ax=None):
        """
        Advanced hbarplot for multiple RFI computations

        Args:

        """
        return fi_sns_gbarplot(self, figsize=figsize, ax=ax)
