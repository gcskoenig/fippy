from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import logging

from rfi.backend.causality.sem import StructuralEquationModel

logger = logging.getLogger(__name__)


@dataclass
class SyntheticExample:
    name: str
    sem: StructuralEquationModel

    @property
    def var_names(self):
        return self.sem.dag.var_names

    def get_train_test_data(self, inputs: Tuple[str], target: str, n_train=10 ** 3, n_test=10 ** 2, seed=None) \
            -> Tuple[np.array, np.array, np.array, np.array]:

        assert all([inp in self.var_names for inp in inputs])
        assert target in self.var_names

        train_seed = seed
        test_seed = 2 * seed if seed is not None else None
        logger.info(f'Sampling train and test data with seeds: {train_seed} and {test_seed}')
        train = self.sem.sample(n_train, seed=train_seed).numpy()
        test = self.sem.sample(n_test, seed=test_seed).numpy()

        inputs_ind = np.searchsorted(self.var_names, inputs)
        target_ind = np.searchsorted(self.var_names, target)

        return train[:, inputs_ind], train[:, target_ind], test[:, inputs_ind], test[:, target_ind]

    def sample_and_save(self, n=10 ** 5, seed=42, save_path='data'):
        logger.info(f'Sampling dataset with seeds: {seed}')
        dataset = self.sem.sample(n, seed=seed).numpy()

        logger.info(f'Saving dataset as {save_path}/{self.name}.csv')
        np.savetxt(f'{save_path}/{self.name}.csv', dataset)

        logger.info(f'Saving DAG plot as {save_path}/{self.name}.png')
        self.sem.dag.plot_dag()
        plt.savefig(f'{save_path}/{self.name}.png')
