import pymc3 as pm
import theano.tensor as tt
from theano import shared
import numpy as np
import scipy as sp
# theano.config.gcc.cxxflags = "-fbracket-depth=16000" # default is 256

class Ornstein_Uhlenbeck(pm.Continuous):
    """
    Ornstein-Uhlenbeck Process
    Parameters
    ----------
    D : tensor
        D > 0, diffusion coefficient
    A : tensor
        A > 0, amplitude of fluctuation <x**2>=A
    delta_t: scalar
        delta_t > 0, time step
    """

    def __init__(self, D=None, A=None, B=None,
                 *args, **kwargs):
        super(Ornstein_Uhlenbeck, self).__init__(*args, **kwargs)
        self.D = D
        self.A = A
        self.B = B

    def logp(self, x):
        D = self.D
        A = self.A
        B = self.B

        x_im1 = x[:-1]
        x_i = x[1:]

        ou_like = pm.Normal.dist(mu=x_im1*B, tau=1.0/A/(1-B**2)).logp(x_i)
        return pm.Normal.dist(mu=0.0,tau=1.0/A).logp(x[0]) + tt.sum(ou_like)

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

    def create_model(self, x=None, mu_D=None, sd_D=None, mu_A=None, sd_A=None, delta_t=None, N=None):
        with pm.Model() as model:
            D = pm.Gamma('D', mu=mu_D, sd=sd_D)
            A = pm.Gamma('A', mu=mu_A, sd=sd_A)

            B = pm.Deterministic('B', pm.exp(-delta_t * D / A))

            path = Ornstein_Uhlenbeck('path',D=D, A=A, B=B, observed=x)
        return model

class LangevinIG(BayesianModel):
    """Bayesian model for a Ornstein-Uhlenback process.
    The model has inputs x, and prior parameters for
    gamma distributions for D and A
    """

    def create_model(self, x=None, aD=None, bD=None, aA=None, bA=None, delta_t=None, N=None):
        with pm.Model() as model:
            D = pm.InverseGamma('D', alpha=aD, beta=bD)
            A = pm.Gamma('A', alpha=aA, beta=bA)

            B = pm.Deterministic('B', pm.exp(-delta_t * D / A))

            path = Ornstein_Uhlenbeck('path',D=D, A=A, B=B, observed=x)
        return model

class LangevinIG2(BayesianModel):
    """Bayesian model for a Ornstein-Uhlenback process.
    The model has inputs x, and prior parameters for
    gamma distributions for D and A
    """

    def create_model(self, x1=None, x2=None, aD=None, bD=None, aA1=None, bA1=None, aA2=None, bA2=None, delta_t=None, N=None):
        with pm.Model() as model:
            D = pm.InverseGamma('D', alpha=aD, beta=bD)
            A1 = pm.Gamma('A1', alpha=aA1, beta=bA1)
            A2 = pm.Gamma('A2', alpha=aA2, beta=bA2)

            B1 = pm.Deterministic('B1', pm.exp(-delta_t * D / A1))
            B2 = pm.Deterministic('B2', pm.exp(-delta_t * D / A2))

            path1 = Ornstein_Uhlenbeck('path1',D=D, A=A1, B=B1, observed=x1)
            path2 = Ornstein_Uhlenbeck('path2', D=D, A=A2, B=B2, observed=x2)
        return model
