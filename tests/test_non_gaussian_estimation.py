from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit, train_test_split
from nflows.distributions import StandardNormal
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import logging
from scipy import integrate
from sklearn.datasets import make_moons

from rfi.backend.cnf import NormalisingFlowEstimator
from rfi.backend.mdn import MixtureDensityNetworkEstimator
from rfi.backend.cnf.transforms import ContextualInvertableRadialTransform, ContextualPointwiseAffineTransform


logging.basicConfig(level=logging.INFO)
EPSABS = 0.01
ASSERT_DECIMAL = 1
DEFAULT_TRANSFORMS = 4 * (ContextualInvertableRadialTransform,) + (ContextualPointwiseAffineTransform,)
ESTIMATOR_CLS = [
    NormalisingFlowEstimator,
    MixtureDensityNetworkEstimator
]

torch.autograd.set_detect_anomaly(True)


class TestNonGaussianEstimators:

    @staticmethod
    def _plot_univariate(y_test, y_sampled):
        idlist = [[i + 1] * len(x) for i, x in enumerate([y_test, y_sampled])]
        df = pd.DataFrame(np.array([np.concatenate([y_test, y_sampled]), np.concatenate(idlist)]).T, columns=['value', 'type'])
        df['type'] = df['type'].map({1: 'original', 2: 'sampled'})
        return sns.displot(df, x='value', rug=True, kind='kde', color='black', hue='type')

    def test_univar_uncond(self):  # =================  Univarite unconditional density with Boston dataset ==================
        for estimator_cls in ESTIMATOR_CLS:
            X, y = load_boston(return_X_y=True)

            # Train/test splits
            y_train, y_test = train_test_split(y, random_state=10)

            estimator = estimator_cls(context_size=0, n_epochs=1000, input_noise_std=0.07, n_components=5,
                                      transform_classes=DEFAULT_TRANSFORMS)
            estimator.fit(train_inputs=y_train, verbose=True, val_inputs=y_test, log_frequency=1000)

            # Density check
            int_res = integrate.quad(lambda x: np.exp(estimator.log_prob(x)), -np.inf, np.inf, limit=100)[0]
            logging.info(f'Integral of density: {int_res}')
            np.testing.assert_almost_equal(int_res, 1.0, ASSERT_DECIMAL)

            # Density plotting
            fig, ax = plt.subplots()
            x = np.linspace(y_train.min() - y_train.std(), y_train.max() + y_train.std(), 1000)
            sns.kdeplot(y_train, ax=ax)
            ax.plot(x, np.exp(estimator.log_prob(x)))
            ax.set_title(f'Uni-variate unconditional density ({estimator_cls.__name__})')
            plt.show()

            # Sampling
            with torch.no_grad():
                y_sampled = estimator.sample(num_samples=len(y_test)).reshape(-1)
            g = self._plot_univariate(y_test, y_sampled)
            g.fig.suptitle(f'Uni-variate unconditional sampling ({estimator_cls.__name__})')
            plt.show()

    def test_univar_cond(self):  # =================  Univarite unconditional density with Boston dataset ==================
        for estimator_cls in ESTIMATOR_CLS:
            X, y = load_boston(return_X_y=True)

            # Train/test splits
            y_train, y_test, X_train, X_test = train_test_split(y, X, random_state=11)

            estimator = estimator_cls(context_size=X_train.shape[1], n_epochs=1000, input_noise_std=0.3, context_noise_std=0.1,
                                      n_components=5, lr=0.001, transform_classes=DEFAULT_TRANSFORMS)
            estimator.fit(train_inputs=y_train, train_context=X_train, verbose=True, val_inputs=y_test, val_context=X_test,
                          log_frequency=1000)

            # Density check
            int_res = integrate.quad_vec(lambda x: np.exp(estimator.log_prob(np.repeat(x, len(X_test)), context=X_test)),
                                         -np.inf, np.inf, epsabs=EPSABS)[0]
            logging.info(f'Mean integral of cond density: {int_res.mean()}')
            np.testing.assert_array_almost_equal(int_res, np.ones(len(X_test)), ASSERT_DECIMAL)

            # Sampling
            with torch.no_grad():
                y_sampled = estimator.sample(num_samples=1, context=X_test).reshape(-1)
            g = self._plot_univariate(y_test, y_sampled)
            g.fig.suptitle(f'Uni-variate conditional sampling ({estimator_cls.__name__})')
            plt.show()

    def test_bivar_uncond(self):  # =================  Bivariate unconditional density with moons dataset ==================
        for estimator_cls in ESTIMATOR_CLS:
            data = make_moons(n_samples=2000, noise=0.1, random_state=41)[0]
            data_train, data_test = train_test_split(data, random_state=42)

            estimator = estimator_cls(inputs_size=2, context_size=0, n_epochs=1000, input_noise_std=0.2, lr=0.02,
                                      base_distribution=StandardNormal(shape=[2]), transform_classes=DEFAULT_TRANSFORMS)
            estimator.fit(train_inputs=data_train, verbose=True, val_inputs=data_test, log_frequency=1000)

            # Density check
            int_res = integrate.dblquad(lambda x, y: np.exp(estimator.log_prob(np.array([x, y]))), -np.inf, np.inf, -np.inf,
                                        np.inf, epsabs=EPSABS)[0]
            logging.info(f'Integral of density: {int_res}')
            np.testing.assert_almost_equal(int_res, 1.0, ASSERT_DECIMAL)

            # Density plotting
            fig, ax = plt.subplots()
            x, y = np.meshgrid(np.linspace(data_train[:, 0].min() - data_train[:, 0].std(),
                                           data_train[:, 0].max() + data_train[:, 0].std(),
                                           200),
                               np.linspace(data_train[:, 1].min() - data_train[:, 1].std(),
                                           data_train[:, 1].max() + data_train[:, 1].std(),
                                           200))
            z = np.exp(estimator.log_prob(np.stack((x.flatten(), y.flatten()), axis=1))).reshape(x.shape)
            z = z[:-1, :-1]
            c = ax.pcolormesh(x, y, z, cmap='Blues', vmin=0.0, vmax=z.max())
            ax.set_title(f'Bi-variate unconditional density ({estimator_cls.__name__})')
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c, ax=ax)

            ax.scatter(data_test[:, 0], data_test[:, 1], zorder=1, alpha=0.1, color='black', label='Test sample')
            fig.legend()
            plt.show()

            # Sampling
            with torch.no_grad():
                data_sampled = np.squeeze(estimator.sample(num_samples=len(data_test)))
            fig, ax = plt.subplots()
            ax.scatter(data_sampled[:, 0], data_sampled[:, 1], alpha=0.25, color='Blue', label='Sampled data')
            ax.scatter(data_test[:, 0], data_test[:, 1], alpha=0.25, color='Black', label='Test data')
            ax.set_title(f'Bi-variate unconditional sampling ({estimator_cls.__name__})')
            fig.legend()
            plt.show()

    def test_bivar_cond(self):  # =================  Bivariate conditional density with moons dataset ==================
        for estimator_cls in ESTIMATOR_CLS:
            y, X = make_moons(n_samples=2000, noise=0.1, random_state=41)
            y_train, y_test, X_train, X_test = train_test_split(y, X, random_state=42)
            X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)

            estimator = estimator_cls(inputs_size=2, context_size=1, n_epochs=1000, input_noise_std=0.2, lr=0.02,
                                      base_distribution=StandardNormal(shape=[2]), transform_classes=DEFAULT_TRANSFORMS,
                                      cat_context=[0])
            estimator.fit(train_inputs=y_train, train_context=X_train, verbose=True, val_inputs=y_test, val_context=X_test,
                          log_frequency=1000)

            # Density check
            int_res = integrate.dblquad(lambda x, y: np.exp(estimator.log_prob(np.array([x, y]), context=np.zeros((1, 1)))),
                                        -np.inf, np.inf, -np.inf, np.inf, epsabs=EPSABS)[0]
            logging.info(f'Integral of conditional density (X = 0): {int_res}')
            np.testing.assert_almost_equal(int_res, 1.0, ASSERT_DECIMAL)
            int_res = integrate.dblquad(lambda x, y: np.exp(estimator.log_prob(np.array([x, y]), context=np.ones((1, 1)))),
                                        -np.inf, np.inf, -np.inf, np.inf, epsabs=EPSABS)[0]
            logging.info(f'Integral of conditional density (X = 0): {int_res}')
            np.testing.assert_almost_equal(int_res, 1.0, ASSERT_DECIMAL)

            # Density plotting
            fig, ax = plt.subplots()
            x, y = np.meshgrid(np.linspace(y_train[:, 0].min() - y_train[:, 0].std(),
                                           y_train[:, 0].max() + y_train[:, 0].std(),
                                           200),
                               np.linspace(y_train[:, 1].min() - y_train[:, 1].std(),
                                           y_train[:, 1].max() + y_train[:, 1].std(),
                                           200))
            inputs = np.stack((x.flatten(), y.flatten()), axis=1)
            z0 = np.exp(estimator.log_prob(inputs, context=np.zeros((len(inputs), 1)))).reshape(x.shape)
            z1 = np.exp(estimator.log_prob(inputs, context=np.ones((len(inputs), 1)))).reshape(x.shape)
            z0 = z0[:-1, :-1]
            z1 = z1[:-1, :-1]
            c0 = ax.pcolormesh(x, y, z0, cmap='Blues', vmin=0.0, vmax=z0.max(), label='X = 0', alpha=0.5)
            c1 = ax.pcolormesh(x, y, z1, cmap='Reds', vmin=0.0, vmax=z1.max(), label='X = 0', alpha=0.25)

            ax.set_title(f'Bi-variate conditional density (0 - blue, 1 - red) ({estimator_cls.__name__})')
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c0, ax=ax)
            fig.colorbar(c1, ax=ax)

            ax.scatter(y_test[:, 0], y_test[:, 1], zorder=1, alpha=0.1, color='Black', label='Test sample')
            fig.legend()
            plt.show()

            # Sampling
            with torch.no_grad():
                y_sampled0 = np.squeeze(estimator.sample(num_samples=1, context=np.zeros((len(y_test) // 2, 1))))
                y_sampled1 = np.squeeze(estimator.sample(num_samples=1, context=np.ones((len(y_test) // 2, 1))))
            fig, ax = plt.subplots()
            ax.scatter(y_sampled0[:, 0], y_sampled0[:, 1], alpha=0.25, color='Blue', label='Sampled data')
            ax.scatter(y_sampled1[:, 0], y_sampled1[:, 1], alpha=0.25, color='Red', label='Sampled data')
            ax.scatter(y_test[:, 0], y_test[:, 1], alpha=0.25, color='Black', label='Test data')
            ax.set_title(f'Bi-variate conditional sampling (0 - blue, 1 - red) ({estimator_cls.__name__})')
            fig.legend()
            plt.show()
