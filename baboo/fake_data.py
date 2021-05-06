import sdeint
import numpy as np

def two_component_fake_data(times, Omgc_start=100,
                            xi_c=1e-11, xi_s=1e-11,
                            N_c=1e-11, N_s=-1e-11,
                            tau_c=1e6, tau_s=3e6,
                            R_c=1e-12, R_s=1e-12):
    """
    Parameters:
    -----------
    """
    tau = (tau_c * tau_s) / (tau_c + tau_s)
    # get start for omega_s
    Omgs_start = Omgc_start - (N_c - N_s) * tau
    # define helper functions for sdeint
    F_int = np.array([[-1/tau_c, 1/tau_c],[1/tau_s, -1/tau_s]])
    def f(x, t):
        return F_int.dot(x) + np.array([N_c, N_s])
    def g(x, t):
        return np.diag([xi_c, xi_s])
    states = sdeint.itoint(f, g, np.array([Omgc_start, Omgs_start]), times)
    data = states.copy()
    data[:, 0] += np.random.randn(times.size) * np.sqrt(R_c)
    data[:, 1] += np.random.randn(times.size) * np.sqrt(R_s)
    return data

def freq_fake_data(times,Ndot,
                   f0,fdot0,
                   xi_f, xi_fdot,
                   R):
    """""
    Parameters:
    -----------
    """""
    #Functions for sdeint
    F_int=np.array([[0,1],[0,0]])
    def f(x,t):
        return F_int.dot(x)+np.array([0,Ndot])
    def g(x,t):
        return np.diag([xi_f,xi_fdot])
    states=sdeint.itoint(f,g,np.array([f0,fdot0]),times)
    data=states.copy()
    data[:,0] += np.random.randn(times.size)*np.sqrt(R)
    data[:,1] += np.random.randn(times.size)*np.sqrt(R)
    return data
