from pathlib import Path

import dynesty
import libstempo
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from dynesty import utils as dyfunc
from scipy.linalg import inv
from numba import jit, njit

def construct_transition(dts):
    F = np.zeros((3, 3, dts.size))
    # \delta phi starts at zero, not at previous phi value for our case!
    F[0, 1, :] = dts
    F[0, 2, :] = dts ** 2 / 2.0
    F[1, 1, :] = 1
    F[1, 2, :] = dts
    F[2, 2, :] = 1
    return F


def construct_torques(dts, fddot):
    T = np.zeros((3, 1, dts.size))
    T[0, 0, :] = dts ** 3 * fddot / 6.0
    T[1, 0, :] = dts ** 2 * fddot / 2.0
    T[2, 0, :] = dts * fddot
    return T


def construct_Q(dts, sig_fdot, sig_f, sig_p):
    Q = np.zeros((3, 3, dts.size))
    Q[0, 0, :] = dts * (
        sig_f ** 2 * dts ** 2 / 3 + sig_fdot ** 2 * dts ** 4 / 20 + sig_p ** 2
    )
    Q[0, 1, :] = dts * (4 * sig_f ** 2 * dts + sig_fdot ** 2 * dts ** 3) / 8
    Q[0, 2, :] = dts ** 3 * sig_fdot ** 2 / 6
    Q[1, 0, :] = Q[0, 1, :]
    Q[1, 1, :] = dts * (sig_f ** 2 + sig_fdot ** 2 * dts ** 2 / 3)
    Q[1, 2, :] = dts ** 2 * sig_fdot ** 2 / 2.0
    Q[2, 0, :] = Q[0, 2, :]
    Q[2, 1, :] = Q[1, 2, :]
    Q[2, 2, :] = dts * sig_fdot ** 2
    return Q

@njit(nopython=True, fastmath=True)
def predict_phi_f_fdot(x, P, transition, torque, process_covariance, nstates):
    # print(x.dtype)
    # print(P.dtype)
    # print(transition.dtype)
    # print(torque.dtype)
    # print(process_covariance.dtype)
    xp = transition @ x + torque
    Pp = transition @ P @ transition.T + process_covariance
    return xp, Pp


@njit(nopython=True, fastmath=True)
def update_phi_f_fdot(xp, Pp, dt_err, measurement_jac):
    innov = np.round(xp[0, 0]) - xp[0, 0]
    innov_covar = measurement_jac @ Pp @ measurement_jac.T + dt_err ** 2 * xp[1, 0] ** 2
    # print(measurement_jac.shape)
    # print((Pp @ measurement_jac.T).shape)
    # print(innov_covar.shape)

    gain = (Pp @ measurement_jac.T) / innov_covar
    x = xp + gain * innov
    P = (np.eye(3) - gain @ measurement_jac) @ Pp
    ll = -0.5 * (np.log(innov_covar) + innov ** 2 / innov_covar + np.log(2 * np.pi))
    return x, P, innov, innov_covar, ll


def run_filter_phi_f_fdot(
    toas, toa_errors, F0, F1, F2, sigp, sigf, sigf_dot, ll_burn=5, efac=1, equad=0
):
    measurement_jac = np.array([1., 0, 0]).reshape((1, 3))

    # time deltas
    deltaTs = np.diff(toas) * 86400

    # toa errors
    new_errors = np.sqrt(toa_errors ** 2 * efac ** 2 + equad)
    deltaT_errors = np.sqrt(new_errors[1:] ** 2 + new_errors[:-1] ** 2)

    # transition, torque, noise matrices
    transitions = construct_transition(deltaTs)
    torques = construct_torques(deltaTs, F2)
    Qs = construct_Q(deltaTs, sigf_dot, sigf, sigp)

    # things we want to keep track of
    ll_tot = 0
    x_final = []
    P_final = []
    innovations = []
    innovation_covariances = []

    ii = 0

    # initial state
    x = np.array([0, F0, F1]).reshape((3, 1))

    # Currently results are surprisingly sensitive to this initial choice.
    # For now we use these defaults
    P = np.diag([1e5 * deltaT_errors[0] ** 2 * F0 ** 2, 5 * F0 ** 2, 5 * F1 ** 2])
    for dt, dt_err in zip(deltaTs, deltaT_errors):
        # print(x.dtype)
        # print(P.dtype)
        # print(transitions.dtype)
        # print(torques.dtype)
        # print(Qs.dtype)

        xp, pp = predict_phi_f_fdot(
            x, P, transitions[:, :, ii], torques[:, :, ii], Qs[:, :, ii], 2
        )
        x, P, innov, innov_cov, ll = update_phi_f_fdot(xp, pp, dt_err, measurement_jac)
        x_final.append(x)
        P_final.append(P)
        # let things settle down
        if ii >= ll_burn and ~np.isnan(ll):
            ll_tot += ll.squeeze()
        innovations.append(innov)
        innovation_covariances.append(innov_cov)
        ii += 1
    if ii == 0:
        ii = -np.inf

    return (
        np.array(x_final).squeeze(),
        np.array(P_final),
        np.array(innovations),
        np.array(innovation_covariances),
        ll_tot,
    )


def run_smoother_phi_f_fdot(
    toas, toa_errors, F0, F1, F2, sigp, sigf, sigf_dot, ll_burn=5, efac=1, equad=0
):
    measurement_jac = np.array([1., 0, 0]).reshape((1, 3))

    # time deltas
    deltaTs = np.diff(toas) * 86400

    # toa errors
    new_errors = np.sqrt(toa_errors ** 2 * efac ** 2 + equad)
    deltaT_errors = np.sqrt(new_errors[1:] ** 2 + new_errors[:-1] ** 2)

    # transition, torque, noise matrices
    transitions = construct_transition(deltaTs)
    torques = construct_torques(deltaTs, F2)
    Qs = construct_Q(deltaTs, sigf_dot, sigf, sigp)

    # things we want to keep track of
    ll_tot = 0
    x_final = []
    P_final = []
    x_predicts = []
    P_predicts = []
    innovations = []
    innovation_covariances = []

    ii = 0

    # initial state
    x = np.array([0, F0, F1]).reshape((3, 1))

    # currently results are surprisingly sensitive to this initial choice.
    P = np.diag([1e5 * deltaT_errors[0] ** 2 * F0 ** 2, 5 * F0 ** 2, 5 * F1 ** 2])
    for dt, dt_err in zip(deltaTs, deltaT_errors):
        xp, pp = predict_phi_f_fdot(
            x, P, transitions[:, :, ii], torques[:, :, ii], Qs[:, :, ii], 2
        )
        x_predicts.append(xp)
        P_predicts.append(pp)
        x, P, innov, innov_cov, ll = update_phi_f_fdot(xp, pp, dt_err, measurement_jac)
        x_final.append(x)
        P_final.append(P)
        # let things settle down
        if ii >= ll_burn and ~np.isnan(ll):
            ll_tot += ll.squeeze()
        innovations.append(innov)
        innovation_covariances.append(innov_cov)
        ii += 1
    if ii == 0:
        ii = -np.inf
    x_final = np.array(x_final).squeeze()
    P_final = np.array(P_final).squeeze()
    x_predicts = np.array(x_predicts).squeeze()
    P_predicts = np.array(P_predicts).squeeze()
    xS = np.zeros(x_final.shape)
    pS = np.zeros(P_final.shape)
    xS[-1, :] = x_final[-1, :]
    pS[-1, :, :] = P_final[-1, :, :]

    for nn in range(0, ii - 1)[::-1]:
        M = P_final[nn, :] @ transitions[:, :, nn + 1].T @ inv(P_predicts[nn + 1, :, :])
        xS[nn, :] = x_final[nn, :] + M @ (xS[nn + 1, :] - x_predicts[nn + 1, :])
        pS[nn, :] = P_final[nn, :] + M @ (pS[nn + 1, :] - P_predicts[nn + 1, :]) @ M.T

    return (
        np.array(xS).squeeze(),
        np.array(pS),
        np.array(innovations),
        np.array(innovation_covariances),
        ll_tot,
    )


def load_utmost_pulsar_and_sample(parfile, timfile, nlive=100, jname="Pulsar"):
    outpath = Path(jname)
    outpath.mkdir(exist_ok=True, parents=True)

    print(f"loading {jname} pulsar")
    pulsar = libstempo.tempopulsar(parfile=parfile, timfile=timfile)
    F0 = float(
        pulsar["F0"].val
        + pulsar["F1"].val * (pulsar.toas()[0] - pulsar["PEPOCH"].val) * 86400
    )
    F1 = float(pulsar["F1"].val)
    F2 = float(pulsar["F2"].val)
    x_final, P_final, innovations, innovations_covars, ll = run_filter_phi_f_fdot(
        pulsar.pets().astype(float), pulsar.toaerrs.astype(float) * 1e-6, float(F0), F1, F2, 0, 0, 0
    )
    param_mins = np.array(
        [
            F0 - 20 * pulsar["F0"].err,
            F1 - 20 * pulsar["F1"].err,
            -1e-20,
            -25,
            -25,
            -60,
            0,
            -30,
            pulsar["PMRA"].val - 3,
            pulsar["PMDEC"].val - 3,
            pulsar["RAJ"].val - 10 * pulsar["RAJ"].err,
            pulsar["DECJ"].val - 10 * pulsar["DECJ"].err,
        ]
    )
    param_maxs = np.array(
        [
            F0 + 20 * pulsar["F0"].err,
            F1 + 20 * pulsar["F1"].err,
            1e-20,
            -5,
            -5,
            -20,
            2,
            0,
            pulsar["PMRA"].val + 3,
            pulsar["PMDEC"].val + 3,
            pulsar["RAJ"].val + 10 * pulsar["RAJ"].err,
            pulsar["DECJ"].val + 10 * pulsar["DECJ"].err,
        ]
    )

    def prior_transform(x):
        return x * (param_maxs - param_mins) + param_mins

    def lnprob(x):
        pulsar["PMRA"].val = x[8]
        pulsar["PMDEC"].val = x[9]
        pulsar["RAJ"].val = x[10]
        pulsar["DECJ"].val = x[11]
        _, _, _, _, ll = run_filter_phi_f_fdot(
            pulsar.pets().astype(float),
            pulsar.toaerrs.astype(float) * 1e-6,
            float(x[0]),
            float(x[1]),
            float(x[2]),
            10 ** x[3],
            10 ** x[4],
            10 ** x[5],
            efac=x[6],
            equad=10 ** x[7],
        )
        if np.isnan(ll):
            return -np.inf
        else:
            return ll

    print(f"making {jname} plots")
    sampler = dynesty.NestedSampler(lnprob, prior_transform, 12, nlive=nlive)
    sampler.run_nested()
    results = sampler.results

    results = sampler.results
    # get equal weighted samples
    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
    # Resample weighted samples.
    samples_equal = dyfunc.resample_equal(samples, weights)

    # fig = corner(temposamps, smooth=True, smooth1d=True, color='C1', label='Temponest')

    fig = corner(
        samples_equal,
        smooth=True,
        smooth1d=True,
        color="C0",
        truth_color="k",
        label="Our method",
        **{"pcolor_kwargs": {"alpha": 0.6}},
        levels=[1 - np.exp(-2)],
    )
    labels = [
        "$\\nu(t_0)$",
        "$\\dot \\nu(t_0)$",
        "$\\ddot{f}$",
        "$\\log_{10}\\sigma_\\phi$",
        "$\\log_{10}\\sigma_f$",
        "$\\log_{10}\\sigma_{\\dot{f}}$",
        "EFAC",
        "$\\log_{10}$EQUAD",
        "$\\mu_\\alpha \\cos{\\delta}$",
        "$\\mu_\\delta$",
        "$\\alpha$",
        "$\\delta$",
    ]
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii, ii].set_title(label)
    plt.savefig(outpath.joinpath(f"{jname}_corner.pdf"))
    plt.show()

    return (
        results,
        samples_equal,
        pulsar.pets(),
        pulsar.toaerrs * 1e-6,
        pulsar.residuals(),
    )
