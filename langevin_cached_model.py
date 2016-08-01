import pymc3 as pm
from theano import shared
import numpy as np
import scipy as sp

class BayesianModel(object):
    samples = 10000

    def __init__(self, cache_model=True):
        self.cached_model = None
        self.cached_start = None
        self.cached_sampler = None
        self.shared_vars = {}

    def cache_model(self, **inputs):
        self.shared_vars = self._create_shared_vars(**inputs)
        self.cached_model = self.create_model(**self.shared_vars)

    def create_model(self, **inputs):
        raise NotImplementedError('This method has to be overwritten.')

    def _create_shared_vars(self, **inputs):
        shared_vars = {}
        for name, data in inputs.items():
            shared_vars[name] = shared(np.asarray(data), name=name)
        return shared_vars

    def run(self, reinit=True, **inputs):
        if self.cached_model is None:
            self.cache_model(**inputs)

        for name, data in inputs.items():
            self.shared_vars[name].set_value(data)

        trace = self._inference(reinit=reinit)
        return trace

    def _inference(self, reinit=True):
        with self.cached_model:
            if reinit or (self.cached_start is None) or (self.cached_sampler is None):
                self.cached_start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
                self.cached_sampler = pm.NUTS(scaling=self.cached_start)

            trace = pm.sample(self.samples, self.cached_sampler, start=self.cached_start)

        return trace


class Langevin(BayesianModel):
    """Bayesian model for a Ornstein-Uhlenback process.
    The model has inputs x, and prior parameters for
    gamma distributions for D and A
    """

    def create_model(self, x=None, mu_D=None, sd_D=None, mu_A=None, sd_A=None, delta_t=0.01):
        with pm.Model() as model:
            D = pm.Gamma('D', mu=mu_D, sd=sd_D)
            A = pm.Gamma('A', mu=mu_A, sd=sd_A)

            S = 1.0 - pm.exp(-2.0 * delta_t * D / A)

            ss = pm.exp(-delta_t * D / A)

            path = pm.Normal('path_0', mu=0.0, tau=1 / A, observed=x[0])
            for i in range(1, 50):
                path = pm.Normal('path_%i' % i,
                                 mu=path * ss,
                                 tau=1.0 / A / S,
                                 observed=x[i])
        return model
