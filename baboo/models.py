"""
Models defined using Kalman Filter in `kalman.py`

List of Models:

    `TwoComponentModel`: Two component model using full solution and
    non-uniform time sampling. THIS IS THE ONE YOU PROBABLY WANT TO USE.

    `OneComponentModel`: Simple one-component model with spin-wandering.

    `SecondSpindownModel`: Include a second spindown and track frequency derivative.
"""
import numpy as np
from .kalman import KalmanFilterTimeVarying, KalmanFilterTimeVaryingOneState
import bilby

class TwoComponentModel(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None,
                 Q=None, transition=None, B=None, solve=True, params=None):
        super(TwoComponentModel, self).__init__(transition, design, Q, measurement_cov, B, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'relax_ratio': None, 'reduced_relax': None,
                      'lag': None, 'omegac_dot': None,
                      'Qc': None, 'Qs': None,
                      'EFAC': None,'EQUAD': None}
        else:
            self.params = params.copy()

    @property
    def transformed_params(self):
        A, B, C, D = self.params['relax_ratio'], (self.params['reduced_relax'])**-1, self.params['lag'], self.params['omegac_dot']
        taus = (1+A) * B
        tauc = (1+A) / A * B
        qc = self.params['Qc']
        qs = self.params['Qs']
        Nc = (D + A/(1+A) * C/B)
        Ns = D - C/B * (1+A)**-1
        return tauc, taus, qc, qs, Nc, Ns, self.params['EFAC'], self.params['EQUAD']

    @property
    def keylist(self):
        return ['relax_ratio', 'reduced_relax',
                   'lag', 'omegac_dot', 'Qc', 'Qs',
                   'EFAC','EQUAD']

    def _check_params(self, params):
        if list(self.params.keys()) != list(params.keys()):
            raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")


    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # check
        self._check_params(params)
        # update
        self.params = params.copy()
        tauc, taus, Qc, Qs, Nc, Ns, EFAC, EQUAD = self.transformed_params
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition(tauc, taus, dts)
        self.update_Q(tauc, taus, Qc, Qs, dts)
        self.update_torque(tauc, taus, Nc, Ns, dts)
        # rescale and add more uncertainty, as done typically with pulsar # timing
        self.R[0, 0, :] = self._R[0, 0, :] * EFAC + EQUAD

    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewha reasonable starting point for
        # initial state. I dont' think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        try:
            self.params = params.copy()
            tauc, taus, Qc, Qs, Nc, Ns, EFAC, EQUAD = self.transformed_params
            lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
            if isinstance(params, dict):
                omgc_0 = params['omgc_0']
                omgs_0 = omgc_0 - lag
            elif isinstance(params, np.ndarray):
                omgc_0 = self.data[0,0]
                omgs_0 = omgc_0 - lag
            return self.ll_on_data(self.data, params, x0=np.array([omgc_0, omgs_0]),
                                   P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e1,
                                   burn=loglikelihood_burn,
                                   return_states=return_states)
        except np.linalg.LinAlgError:
            return -np.inf
    def update_Q(self, tauc, taus, Qc, Qs, dts):
        tauc = np.real(tauc)
        taus = np.real(taus)
        Qc = np.real(Qc)
        Qs = np.real(Qs)
        dts = np.real(dts)
        # useful values
        const = (1/tauc + 1/taus)
        tauc_plus_taus_squared = (tauc + taus)**2
        expvals = np.exp(-const * dts).real
        Q = np.zeros((2, 2, dts.size))
        tau = const**-1
        Q[0,0,:] = (Qc * tauc**2 + Qs*taus**2) * dts +\
                 (2 * Qc *tauc * taus - 2 * Qs * taus**2) * tau * (1 - expvals) +\
                 (Qc * taus**2 + Qs * taus**2) * (1 - expvals**2) * (tau / 2)

        Q[1,1,:] = (Qc * tauc**2 + Qs*taus**2) * dts +\
                (2 * Qs *tauc * taus - 2 * Qc * tauc**2) * tau * (1 - expvals) +\
                (Qc * tauc**2 + Qs * tauc**2) * (1 - expvals**2) * (tau / 2)

        Q[1,0,:] = (Qs*taus**2 + Qc*tauc**2) * dts + \
                (Qc * tauc * taus - tauc**2*Qc + Qs*tauc*taus - Qs*taus**2)*tau*(1 - expvals) - \
                (Qc + Qs)*(tauc*taus)*(1-expvals**2)*(tau/2)
        Q[0,1,:]  = Q[1, 0, :].copy()
        self.Q =  Q / (tauc_plus_taus_squared)

            # exponential transition matrix
    def update_transition(self, tauc, taus, dts):
        tau = (tauc * taus) / (tauc + taus)
        expvals = np.exp(-dts / tau)
        transition = np.zeros((2,2,dts.size))
        transition[0, 0, :] = tauc + taus * expvals
        transition[0, 1, :] = taus - taus * expvals
        transition[1, 0, :] = tauc - tauc * expvals
        transition[1, 1, :] = taus + tauc * expvals
        self.transition = transition / (tauc + taus)

    def update_torque(self, tauc, taus, Nc, Ns, dts):
        tau = (tauc * taus) / (tauc + taus)
        tauc_plus_taus_squared = (tauc + taus)**2
        expvals = np.exp(-dts / tau)
        omg_dot = (tauc * Nc + taus * Ns) / (tauc + taus)
        torques = np.zeros((2, dts.size))

        torques[0, :] = omg_dot * dts + \
                        tau**2 * (tauc**-1 * (Nc - Ns) * (1 - expvals))
        torques[1, :] = omg_dot * dts + \
                        tau**2 * (taus**-1 * (Ns - Nc) * (1 - expvals))
        self.B = torques

class OneComponentModel(KalmanFilterTimeVaryingOneState):
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, Q=None,
                 transition=None, B=None):
        super(OneComponentModel, self).__init__(transition, design, Q,
                                                measurement_cov, B)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self._R = self.R.copy()

    def param_map(self, params):
        # Remember to adjust the param labels in the sample.py file
        Q, N = params['sigma2'], params['torque']
        return Q, N, params['EFAC'], params['EQUAD']

    @property
    def keylist(self):
        # return ['sigma2', 'torque', 'omgc0', 'EFAC', 'EQUAD']
        return ['sigma2', 'torque', 'omgc_0', 'EFAC', 'EQUAD']

    def _check_params(self, params):
        # print(list(params.keys()))
        if self.keylist != list(params.keys()):
            raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")


    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        self._check_params(params)
        Q, N, EFAC, EQUAD = self.param_map(params)
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition_fast(dts)
        self.update_Q_fast(Q, dts)
        self.update_torques_fast(N, dts)
        # self.R = self.R * EFAC + EQUAD
        self.R = self._R * EFAC + EQUAD

    def update_transition_fast(self, dts):
        transition = np.ones((dts.size, 1, 1))
        self.transition = transition.copy()

    def update_Q_fast(self, Q, dts):
        Qs = np.zeros((dts.size, 1, 1))
        Qs[:,0, 0] = Q*dts
        self.Q = Qs.copy()

    def update_torques_fast(self, N, dts):
        torques = np.zeros((dts.size, 1, 1))
        torques[:,0, 0] = N*dts
        self.B = torques.copy()


    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        if self.param_map is None:
            # assume params are the three params we want.
            Q, N, EFAC, EQUAD = params
        else:
            Q, N, EFAC, EQUAD = self.param_map(params)
        if isinstance(params, dict):
            omgc_0 = params['omgc_0']
        elif isinstance(params, list):
            omgc_0 = params[-1]
        x = self.ll_on_data(self.data, params, x0=np.array([omgc_0]),
                            P0=np.eye(self.nstates) * np.max(self.R)*1e1,
                            burn=loglikelihood_burn,
                            return_states=return_states)
        return x

class SecondSpindownModel(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, Q=None,
                 transition=None, B=None, solve=True, params=None):
        super(SecondSpindownModel, self).__init__(transition, design, Q, measurement_cov, B, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'Q1': None, 'Q2':None, 'N2': None, 'EFAC': None, 'EQUAD': None}
        else:
            self.params = params


    @property
    def transformed_params(self):
        return self.params['Q1'],self.params['Q2'],self.params['N2'], self.params['EFAC'], self.params['EQUAD']

    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # update parameters
        self.params = params.copy()
        # get transformed version
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition(dts)
        self.update_Q(Q1, Q2, dts)
        self.update_torques(N2, dts)
        self.R = self._R * EFAC + EQUAD

    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.ll_on_data(self.data, params, x0=np.array([omg0, omgdot0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               burn=loglikelihood_burn,
                               return_states=return_states)
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.run_smoother(self.data, params, x0=np.array([omg0, omgdot0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               )


    def update_transition(self, dts):
        transition = np.zeros((2, 2, dts.size))
        transition[0, 0, :] = 1
        transition[1, 1, :] = 1
        transition[0, 1, :] = dts
        self.transition = transition


    def update_Q(self, Q1, Q2, dts):
        Qs = np.zeros((2, 2, dts.size))
        Qs[0, 0, :] = (Q1**2)*dts+(Q2**2)*dts**3/3
        Qs[1, 1, :] = (Q2**2)*dts
        Qs[0, 1, :] = ((Q2*dts)**2)/2
        Qs[1, 0, :] = ((Q2*dts)**2)/2
        self.Q = Qs

    def update_torques(self, N, dts):
        torques = np.zeros((2, dts.size))
        torques[0, :] = N*dts**2 / 2
        torques[1, :] = N*dts
        self.B = torques
