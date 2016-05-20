import numpy as np
from itertools import accumulate

def time_series(k,ga,diff,delta_t,N,G):
    # differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
    def next_point(x, y):
        amplitude = np.sqrt(2 * diff * delta_t / float(G))
        return x - k / ga * x * delta_t / float(G) + amplitude * y

    # random force
    w=np.random.normal(0,1,N*G)
    xx = np.fromiter(accumulate(w, next_point), np.float)

    return xx[::G]  # only use every G point

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
# using 4th-order Runge-Kutta
def next_point_RK4(x,y):
    k0=-a*x
    k1=-a*(x+k0/2.0)
    k2=-a*(x+k1/2.0)
    k3=-a*(x+k2)
    return x + (k0+2*k1+2*k2+k3)/6 + ampl*y
