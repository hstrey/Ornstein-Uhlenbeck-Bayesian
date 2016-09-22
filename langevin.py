import numpy as np
from itertools import accumulate

def time_series_sim(k,ga,diff,delta_t,N,G):
    """ returns a time series that is the solution of a Langevin equation describing a Brownian particle in a harmonic potential
    :param k: Spring constant
    :param ga: friction coefficient
    :param diff: Diffusion coefficient
    :param delta_t: time step
    :param N: number of samples that are returned
    :param G: ratio of simulated data points and returned samples (100 means that only every 100th point is used)
    :return:
    """
    # differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
    def next_point(x, y):
        amplitude = np.sqrt(2 * diff * delta_t / float(G))
        return x - k / ga * x * delta_t / float(G) + amplitude * y

    # random force
    w=np.random.normal(0,1,N*G)
    xx = np.fromiter(accumulate(w, next_point), np.float)

    return xx[::G]  # only use every G point

def time_series(A,D,delta_t,N):
    """ returns a time series that is the solution of a Langevin equation describing a Brownian particle in a harmonic potential
    :param A: mean square amplitude - oscillator strength
    :param D: Diffusion coefficient
    :param delta_t: time step
    :param N: number of samples that are returned
    :return:
    """
    # using Smolukowski solution for simulation

    #first point
    x=[np.random.normal(0,np.sqrt(A))]
    stddev = np.sqrt(A * (1.0 - np.exp(-2.0 * D / A * delta_t)))

    for i in range(N-1):
        x.append(np.random.normal(x[-1]*np.exp(-D/A*delta_t),stddev))

    return np.array(x)

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
# using 4th-order Runge-Kutta
def next_point_RK4(x,y):
    k0=-a*x
    k1=-a*(x+k0/2.0)
    k2=-a*(x+k1/2.0)
    k3=-a*(x+k2)
    return x + (k0+2*k1+2*k2+k3)/6 + ampl*y
