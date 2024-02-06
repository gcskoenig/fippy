from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
import pandas as pd
import logging

from fippy.backend.causality.sem import StructuralEquationModel
from fippy.utils import search_nonsorted

logger = logging.getLogger(__name__)


@dataclass
class SyntheticExample:
    sem: StructuralEquationModel
    name: str = None

    @property
    def var_names(self):
        return self.sem.dag.var_names

    def get_train_test_data(self, context_vars: Tuple[str], target_var: str, n_train=10 ** 3, n_test=10 ** 2, seed=None,
                            as_dataframes=False, **kwargs) -> \
            Union[Tuple[np.array, np.array, np.array, np.array], Tuple[pd.DataFrame, pd.DataFrame]]:

        assert all([inp in self.var_names for inp in context_vars])
        assert target_var in self.var_names

        train_seed = seed
        test_seed = 2 * seed if seed is not None else None
        logger.info(f'Sampling {n_train} train observations and {n_test} test observations '
                    f'with seeds: {train_seed} and {test_seed}')
        train = self.sem.sample(n_train, seed=train_seed).numpy()
        test = self.sem.sample(n_test, seed=test_seed).numpy()

        inputs_ind = search_nonsorted(self.var_names, context_vars)
        target_ind = search_nonsorted(self.var_names, target_var)
        if not as_dataframes:
            return train[:, inputs_ind], train[:, target_ind], test[:, inputs_ind], test[:, target_ind]
        else:
            train_df = pd.DataFrame(train[:, inputs_ind], columns=context_vars)
            train_df[target_var] = train[:, target_ind]

            test_df = pd.DataFrame(test[:, inputs_ind], columns=context_vars)
            test_df[target_var] = test[:, target_ind]
            return train_df, test_df

    def sample_and_save(self, n=10 ** 5, seed=42, save_path='data'):
        logger.info(f'Sampling dataset with seeds: {seed}')
        dataset = self.sem.sample(n, seed=seed).numpy()

        file_name = self.sem.name if self.name is None else self.name
        logger.info(f'Saving dataset as {save_path}/{file_name}.csv')
        np.savetxt(f'{save_path}/{self.name}.csv', dataset)

        logger.info(f'Saving DAG plot as {save_path}/{file_name}.png')
        self.sem.dag.plot_dag()
        plt.savefig(f'{save_path}/{self.name}.png')
