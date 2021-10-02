import kalman_toas as kt
import numpy as np
import numpy.testing as npt


DT_VAL = 1.123
DTS = np.array([DT_VAL])
FDOT = -1.2345e-10
SIG_PHI = 0.113
SIG_F = 0.00978
SIG_F1 = 0.00011


def test_construct_transition():
    func = kt.construct_transition(DTS)
    true = np.array([[0, DT_VAL, DT_VAL**2 / 2],
                     [0, 1, DT_VAL],
                     [0, 0, 1]])
    npt.assert_array_equal(func.squeeze(), true)


def test_construct_torques():
    func = kt.construct_torques(DTS, FDOT)
    true = np.array([FDOT * DT_VAL**3 / 6.,
                     FDOT * DT_VAL**2 / 2.,
                     FDOT * DT_VAL])
    npt.assert_array_equal(func.squeeze(), true)
    npt.assert_array_equal(func.squeeze(), true)


def test_construct_Q():
    func = kt.construct_Q(DTS, np.sqrt(SIG_F1), 0, 0)
    true_Q_fdot = np.array([[SIG_F1 * DT_VAL**5 / 20, SIG_F1 * DT_VAL**4 / 8, SIG_F1 * DT_VAL**3 / 6],
                            [SIG_F1 * DT_VAL**4 / 8, SIG_F1 * DT_VAL
                             ** 3 / 3, SIG_F1 * DT_VAL**2 / 2],
                            [SIG_F1 * DT_VAL**3 / 6, SIG_F1 * DT_VAL**2 / 2, SIG_F1 * DT_VAL]])
    npt.assert_array_almost_equal(func.squeeze(), true_Q_fdot)
    func = kt.construct_Q(DTS, 0, np.sqrt(SIG_F), 0)
    true_Q_f = np.array([[SIG_F * DT_VAL**3 / 3, SIG_F * DT_VAL**2 / 2, 0],
                         [SIG_F * DT_VAL**2 / 2, SIG_F * DT_VAL, 0],
                         [0, 0, 0]])
    npt.assert_array_almost_equal(func.squeeze(), true_Q_f)
    func = kt.construct_Q(DTS, 0, 0, np.sqrt(SIG_PHI))
    true_Q_p = np.array([[SIG_PHI * DT_VAL, 0, 0], [0, 0, 0], [0, 0, 0]])
    npt.assert_array_almost_equal(func.squeeze(), true_Q_p)

    func = kt.construct_Q(DTS, np.sqrt(
        SIG_F1), np.sqrt(SIG_F), np.sqrt(SIG_PHI))
    npt.assert_array_almost_equal(
        func.squeeze(), true_Q_f + true_Q_p + true_Q_fdot)
    npt.assert_array_almost_equal(
        func.squeeze(), true_Q_f + true_Q_p + true_Q_fdot)
