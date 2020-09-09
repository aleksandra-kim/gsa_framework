import numpy as np

def eFAST_first_order(Y, N, M, omega):
    """Sobol first order index estimator."""
    f = np.fft.fft(Y)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    D1 = 2 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
    return D1 / V

def eFAST_total_order(Y, N, omega):
    """Sobol total order index estimator."""
    f = np.fft.fft(Y)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    Dt = 2 * sum(Sp[np.arange(int(omega / 2))])
    return (1 - Dt / V)

def eFAST_indices(dict_):
    """Compute estimations of Sobol' first and total order indices.

    High values of the Sobol first order index signify important parameters, while low values of the  total indices
    point to non-important parameters. First order computes main effects only, total order takes into account
    interactions between parameters.

    Parameters
    ----------
    dict_ : dict
        Dictionary that contains model outputs ``y`` obtained by running model on Saltelli samples,
        number of Monte Carlo iterations ``iterations``, and number of parameters ``num_params``.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices.

    References
    ----------
    Paper:
        A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output.
        Saltelli A., Tarantola S., Chan K. P.-S.
        https://doi.org/10.1080/00401706.1999.10485594
    Link with the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/analyze/fast.py

    TODO what to do with M? should be consistent in both sampling and FAST indices computation

    """

    y = dict_.get('y')
    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    # Recreate the vector omega used in the sampling
    from sampling.get_samples import get_omega_eFAST
    M = 4
    omega = get_omega_eFAST(num_params, iterations, M)
    # Calculate and Output the First and Total Order Values
    first = np.zeros(num_params)
    total = np.zeros(num_params)
    first[:], total[:] = np.nan, np.nan
    for i in range(num_params):
        l = np.arange(i * iterations, (i + 1) * iterations)
        first[i] = eFAST_first_order(y[l], iterations, M, omega[0])
        total[i] = eFAST_total_order(y[l], iterations, omega[0])
    sa_dict = {
        'eFAST_first': first,
        'eFAST_total': total,
    }
    return sa_dict