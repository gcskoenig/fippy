from rfi.decorrelators.decorrelator import Decorrelator
import logging

logger = logging.getLogger(__name__)


class NaiveDecorrelator(Decorrelator):
    """
    Naive implementation that just independently resamples
    the variables to be dropped.
    """
    def __init__(self, X_train, sampler, X_val=None, **kwargs):
        self.sampler = sampler
        super().__init__(X_train, X_val=True)

    def train(self, K, J, C):
        """
        Trains sampler using dataset to resample variable jj relative to G.
        Args:
            K: features to be knocked out from J cup C to C
            J: features on top of baseline
            C: baseline conditioning set
        """
        K = Decorrelator._to_array(Decorrelator._order_fset(K))
        J = Decorrelator._to_array(Decorrelator._order_fset(J))
        C = Decorrelator._to_array(Decorrelator._order_fset(C))
        super().train(K, J, C)

        if len(K) > 0:
            self.sampler.train(K, C)

        def decorrelationfunc(X_test):
            values_test = X_test.copy()

            # index in K that are in J can directly be resampled using C
            if len(K) > 0:
                smpl = self.sampler.sample(X_test, K, C)
                values_test.loc[:, K] = smpl.to_numpy()

            return values_test  # pandas dataframe

        self._store_decorrelationfunc(K, J, C,
                                      decorrelationfunc)

        return None
