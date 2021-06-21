import numpy as np
from .model_base import ModelBase
from ..utils import uniform_rescale

# TODO should we keep self.num_params in each model?
# TODO implement K-function, also called Bratley: http://www.sfu.ca/~ssurjano/bratleyetal92.html
#      it's also in the Saltelli paper, 2009


######################
# ## Test functions ###
# #####################
class Morris(ModelBase):
    """Class that implements the Morris function.

    Parameters
    ----------
    num_params : int
        Number of model inputs.
    num_influential : int
        Number of influential inputs.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Paper:
        :cite:ts:`morris2006sampling`
    Useful link:
        http://www.sfu.ca/~ssurjano/morretal06.html (there is a typo in the formula, trust the paper)

    """

    def __init__(self, num_params=100, num_influential=5):

        assert num_influential <= num_params
        self.influential_params = np.arange(
            num_influential
        )  # we already know for this function, for comparing with GSA results

        alpha = np.sqrt(12) - 6 * np.sqrt(0.1 * (num_influential - 1))
        beta = 12 * np.sqrt(0.1 * (num_influential - 1))

        self.num_params = num_params
        self.num_influential = num_influential
        self.alpha = alpha
        self.beta = beta
        self.S_dict_analytical = self.get_sensitivity_indices()
        self.S_boolean = self.get_boolean_indices()

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        y = np.zeros(X.shape[0])
        y[:] = self.alpha * np.sum(X[:, : self.num_influential], axis=1)
        for i in range(self.num_influential - 1):
            for j in range(i + 1, self.num_influential):
                y[:] += self.beta * X[:, i] * X[:, j]
        return y

    def get_variance_Y(self):
        """Computes analytical variance of the model output."""

        # Let $ summand1 = \alpha \sum_{i=1}^k x_i $
        # and $ summand2 = \beta \sum_{i=1}^{k-1} \sum_{j=i}^k x_i x_j $
        # Then Morris function $ Y = summand1 + summand2 $
        # 1. Expectation of $Y$:
        E_Y_ = (
            1 / 2 * self.alpha * self.num_influential
            + 1 / 8 * self.beta * self.num_influential * (self.num_influential - 1)
        )
        # 2. Variance of $Y$:
        # 2.1. First we compute th expectation of $Y^2$
        E_summand1_ = (
            1
            / 12
            * self.alpha ** 2
            * self.num_influential
            * (3 * self.num_influential + 1)
        )
        E_2_summand1_summand2_ = (
            2
            * self.alpha
            * self.beta
            * 1
            / 48
            * self.num_influential
            * (self.num_influential - 1)
            * (3 * self.num_influential + 2)
        )
        E_summand2_ = (
            1
            / 2 ** 6
            / 9
            * self.beta ** 2
            * self.num_influential
            * (self.num_influential - 1)
            * (9 * self.num_influential ** 2 + 3 * self.num_influential - 10)
        )
        E_Y2_ = E_summand1_ + E_2_summand1_summand2_ + E_summand2_
        # 2.2. Then the variance becomes:
        Var_Y_ = E_Y2_ - (E_Y_) ** 2
        return Var_Y_

    def get_first_order(self, Var_Y_):
        """Computes analytical first order Sobol index."""
        # Expectation of Y when Xi is fixed to some xi
        # E[Y|Xi=xi] = summand3 + constant*xi, where
        summand3 = 1 / 2 * self.alpha * (
            self.num_influential - 1
        ) + 1 / 8 * self.beta * (self.num_influential - 1) * (self.num_influential - 2)
        constant = self.alpha + 1 / 2 * (self.num_influential - 1) * self.beta
        # 3.2. Now we need to compute variance of this expectation wrt different xi's
        # E_E_Y_given_xi_ = summand3 + 1 / 2 * constant
        # E_E_Y_given_xi_2_ = summand3 ** 2 + summand3 * constant + 1 / 3 * constant ** 2
        # Var_E_Y_given_xi_ = E_E_Y_given_xi_2_ - E_E_Y_given_xi_ ** 2
        Var_E_Y_given_xi_ = (
            1 / 12 * (self.alpha + 1 / 2 * self.beta * (self.num_influential - 1)) ** 2
        )
        S1_value = Var_E_Y_given_xi_ / Var_Y_
        S1 = np.hstack(
            [
                np.ones(self.num_influential) * S1_value,
                np.zeros(self.num_params - self.num_influential),
            ]
        )
        return S1

    def get_total_order(self, Var_Y_):
        """Computes analytical total order Sobol index."""
        a = (
            1 / 2 * self.alpha ** 2
            + 1 / 12 * self.alpha * self.beta * (self.num_influential - 1)
            + 1
            / 144
            * self.beta ** 2
            * (self.num_influential - 1)
            * (3 * self.num_influential - 2)
        )
        ST_value = a / Var_Y_
        ST = np.hstack(
            [
                np.ones(self.num_influential) * ST_value,
                np.zeros(self.num_params - self.num_influential),
            ]
        )
        return ST

    def get_sensitivity_indices(self):
        """Computes analytical first and total order Sobol indices."""
        Var_Y_ = self.get_variance_Y()
        first = self.get_first_order(Var_Y_)
        total = self.get_total_order(Var_Y_)
        dict_ = {
            "First order": first,
            "Total order": total,
        }
        return dict_

    def get_boolean_indices(self):
        """Returns boolean array with ``True`` values for known influential inputs, and ``False`` - for non-influential."""
        S_boolean = np.hstack(
            [
                np.ones(self.num_influential),
                np.zeros(self.num_params - self.num_influential),
            ]
        )
        return S_boolean


class Morris4(ModelBase):
    """Class that implements the modified Morris function that can have 4 levels of input importances.

    Parameters
    ----------
    num_params : int
        Number of model inputs.
    num_influential : int
        Number of influential inputs.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Papers:
        :cite:ts:`kim2021robust`
    Useful link:
        http://www.sfu.ca/~ssurjano/morretal06.html (there is a typo in the formula, trust the paper)

    """

    def __init__(self, num_params=100, num_influential=10):

        assert num_influential <= num_params
        self.influential_params = np.arange(
            num_influential
        )  # we already know for this function, for comparing with GSA results

        self.num_params = num_params
        self.num_influential = num_influential
        self.morris = Morris(self.num_influential, self.num_influential)
        self.alpha = self.morris.alpha
        self.beta = self.morris.beta
        # level of influence
        self.level0_const = 1
        self.level1_const = 1 / np.sqrt(2)
        self.level2_const = 1 / np.sqrt(10)
        self.S_dict_analytical = self.get_sensitivity_indices()
        self.S_boolean = self.get_boolean_indices()

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        k = self.num_influential
        y_level0 = self.level0_const * self.morris(X[:, 0:k])
        y_level1 = self.level1_const * self.morris(X[:, k : 2 * k])
        y_level2 = self.level2_const * self.morris(X[:, 2 * k : 3 * k])
        y = y_level0 + y_level1 + y_level2
        return y

    def get_sensitivity_indices(self):
        """Computes analytical first and total order Sobol indices."""
        Var_Y_ = self.morris.get_variance_Y() * (
            self.level0_const ** 2 + self.level1_const ** 2 + self.level2_const ** 2
        )
        first = np.hstack(
            [
                self.level0_const ** 2 * self.morris.get_first_order(Var_Y_),
                self.level1_const ** 2 * self.morris.get_first_order(Var_Y_),
                self.level2_const ** 2 * self.morris.get_first_order(Var_Y_),
                np.zeros(self.num_params - 3 * self.num_influential),
            ]
        )
        total = np.hstack(
            [
                self.level0_const ** 2 * self.morris.get_total_order(Var_Y_),
                self.level1_const ** 2 * self.morris.get_total_order(Var_Y_),
                self.level2_const ** 2 * self.morris.get_total_order(Var_Y_),
                np.zeros(self.num_params - 3 * self.num_influential),
            ]
        )
        dict_ = {
            "First order": first,
            "Total order": total,
        }
        return dict_

    def get_boolean_indices(self):
        """Returns boolean array with ``True`` values for known influential inputs, and ``False`` - for lowly- and non-influential."""
        S_boolean = np.hstack(
            [
                np.ones(2 * self.num_influential),
                np.zeros(self.num_params - 2 * self.num_influential),
            ]
        )
        return S_boolean


class Borehole(ModelBase):
    """Class that implements the Borehole function.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Original paper:
        :cite:ts:`harper1983sensitivity` -> here the function is slightly different than in the below paper
    Other paper:
        :cite:ts:`moon2012two`
    Useful link:
        http://www.sfu.ca/~ssurjano/borehole.html

    """

    def __init__(self):
        self.params = {
            "rw": [0.05, 0.15],  # radius of borehole (m)
            "r": [100, 50000],  # radius of influence (m)
            "Tu": [63070, 115600],  # transmissivity of upper aquifer (m2/yr)
            "Hu": [990, 1110],  # potentiometric head of upper aquifer (m)
            "Tl": [63.1, 116],  # transmissivity of lower aquifer (m2/yr)
            "Hl": [700, 820],  # potentiometric head of lower aquifer (m)
            "L": [1120, 1680],  # length of borehole (m)
            "Kw": [9855, 12045],  # hydraulic conductivity of borehole (m/yr)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array(
            [3, 5]
        )  # TODO check correcteness in the literature

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):
        rw = X[:, 0]
        r = X[:, 1]
        Tu = X[:, 2]
        Hu = X[:, 3]
        Tl = X[:, 4]
        Hl = X[:, 5]
        L = X[:, 6]
        Kw = X[:, 7]
        # Response is water flow rate (m3/yr)
        y = (
            2
            * np.pi
            * Tu
            * (Hu - Hl)
            / (
                np.log(r / rw)
                * (1 + 2 * L * Tu / (np.log(r / rw) * rw ** 2 * Kw) + Tu / Tl)
            )
        )
        return y


class Wingweight(ModelBase):
    """Class that implements the Wing weight function.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Original paper:
        :cite:ts:`forrester2008engineering`
    Useful link:
        http://www.sfu.ca/~ssurjano/wingweight.html

    """

    def __init__(self):
        self.params = {
            "Sw": [150, 200],  # wing area (ft2)
            "Wfw": [220, 300],  # weight of fuel in the wing (lb)
            "A": [6, 10],  # aspect ratio
            "Lam": [-10, 10],  # quarter-chord sweep (degrees)
            "q": [16, 45],  # dynamic pressure at cruise (lb/ft2)
            "lam": [0.5, 1],  # taper ratio
            "tc": [0.08, 0.18],  # aerofoil thickness to chord ratio
            "Nz": [2.5, 6],  # ultimate load factor
            "Wdg": [1700, 2500],  # flight design gross weight (lb)
            "Wp": [0.025, 0.08],  # paint weight (lb / ft2)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array(
            [0, 9]
        )  # TODO check correcteness in the literature

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):
        Sw = X[:, 0]
        Wfw = X[:, 1]
        A = X[:, 2]
        Lam = X[:, 3] / 180 * np.pi  # convert to radian
        q = X[:, 4]
        lam = X[:, 5]
        tc = X[:, 6]
        Nz = X[:, 7]
        Wdg = X[:, 8]
        Wp = X[:, 9]

        y = (
            0.036
            * Sw ** 0.758
            * Wfw ** 0.0035
            * (A / np.cos(Lam) ** 2) ** 0.6
            * q ** 0.006
            * lam ** 0.04
            * (100 * tc / np.cos(Lam)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )

        return y


class OTLcircuit(ModelBase):
    """Class that implements the OTL circuit function.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.s

    References
    ----------
    Original paper:
        :cite:ts:`ben2007modeling`
    Useful link:
        http://www.sfu.ca/~ssurjano/otlcircuit.html

    """

    def __init__(self):
        self.params = {
            "Rb1": [50, 150],  # resistance b1 (K-Ohms)
            "Rb2": [25, 70],  # resistance b2 (K-Ohms)
            "Rf": [0.5, 3],  # resistance f (K-Ohms)
            "Rc1": [1.2, 2.5],  # resistance c1 (K-Ohms)
            "Rc2": [0.25, 1.2],  # resistance c2 (K-Ohms)
            "beta": [50, 300],  # current gain (Amperes)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array(
            [0, 1, 2, 3]
        )  # TODO check correcteness in the literature

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):
        Rb1 = X[:, 0]
        Rb2 = X[:, 1]
        Rf = X[:, 2]
        Rc1 = X[:, 3]
        Rc2 = X[:, 4]
        beta = X[:, 5]

        Vb1 = 12 * Rb2 / (Rb1 + Rb2)
        temp = beta * (Rc2 + 9) + Rf

        y = (
            (Vb1 + 0.74) * beta * (Rc2 + 9) / temp
            + 11.35 * Rf / temp
            + 0.74 * Rf * beta * (Rc2 + 9) / temp / Rc1
        )

        return y


class Piston(ModelBase):
    """Class that implements the Piston simulation function.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Original paper:
        :cite:ts:`ben2007modeling`
    Useful link:
        http://www.sfu.ca/~ssurjano/piston.html

    """

    def __init__(self):
        self.params = {
            "M": [30, 60],  # piston weight (kg)
            "S": [0.005, 0.020],  # piston surface area (m2)
            "V0": [0.002, 0.010],  # initial gas volume (m3)
            "k": [1000, 5000],  # spring coefficient (N/m)
            "P0": [90000, 110000],  # atmospheric pressure (N/m2)
            "Ta": [290, 296],  # ambient temperature (K)
            "T0": [340, 360],  # filling gas temperature (K)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array(
            [0, 2, 4, 5, 6]
        )  # TODO check correcteness in the literature

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):
        M = X[:, 0]
        S = X[:, 1]
        V0 = X[:, 2]
        k = X[:, 3]
        P0 = X[:, 4]
        Ta = X[:, 5]
        T0 = X[:, 6]

        A = P0 * S + 19.62 * M - k * V0 / S
        V = S / 2 / k * (np.sqrt(A ** 2 + 4 * k * P0 * V0 / T0 * Ta) - A)

        y = 2 * np.pi * np.sqrt(M / (k + S ** 2 * P0 * V0 / T0 * Ta / V ** 2))

        return y


class Moon(ModelBase):
    """Class that implements the Moon function.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Original paper:
        :cite:ts:`moon2012two`
    Useful link:
        http://www.sfu.ca/~ssurjano/moonetal12.html

    """

    def __init__(self, num_dummy=29):
        self.functions_indices = dict(  # left included, right excluded
            borehole=[0, 8],
            wingweight=[8, 18],
            otlcircuit=[18, 24],
            piston=[24, 31],
            dummy=[31, 31 + num_dummy],
        )
        self.borehole, self.wingweight, self.otlcircuit, self.piston = (
            Borehole(),
            Wingweight(),
            OTLcircuit(),
            Piston(),
        )
        self.params = {
            **self.borehole.params,
            **self.wingweight.params,
            **self.otlcircuit.params,
            **self.piston.params,
        }
        self.num_dummy = num_dummy
        if self.num_dummy:
            self.params.update({str(k): [0, 1] for k in range(num_dummy)})
        self.num_params = len(self.params)
        self.influential_params = np.hstack(
            [
                self.borehole.influential_params
                + self.functions_indices["borehole"][0],
                self.wingweight.influential_params
                + self.functions_indices["wingweight"][0],
                self.otlcircuit.influential_params
                + self.functions_indices["otlcircuit"][0],
                self.piston.influential_params + self.functions_indices["piston"][0],
            ]
        )
        self.influential_params.sort()

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        X1 = X[
            :,
            self.functions_indices["borehole"][0] : self.functions_indices["borehole"][
                1
            ],
        ]
        X2 = X[
            :,
            self.functions_indices["wingweight"][0] : self.functions_indices[
                "wingweight"
            ][1],
        ]
        X3 = X[
            :,
            self.functions_indices["otlcircuit"][0] : self.functions_indices[
                "otlcircuit"
            ][1],
        ]
        X4 = X[
            :, self.functions_indices["piston"][0] : self.functions_indices["piston"][1]
        ]
        Xdummy = X[
            :, self.functions_indices["dummy"][0] : self.functions_indices["dummy"][1]
        ]
        X1_rescaled = self.borehole.rescale(X1)
        X2_rescaled = self.wingweight.rescale(X2)
        X3_rescaled = self.otlcircuit.rescale(X3)
        X4_rescaled = self.piston.rescale(X4)
        X = np.hstack([X1_rescaled, X2_rescaled, X3_rescaled, X4_rescaled, Xdummy])
        return X

    def __call__(self, X):
        X1 = X[
            :,
            self.functions_indices["borehole"][0] : self.functions_indices["borehole"][
                1
            ],
        ]
        X2 = X[
            :,
            self.functions_indices["wingweight"][0] : self.functions_indices[
                "wingweight"
            ][1],
        ]
        X3 = X[
            :,
            self.functions_indices["otlcircuit"][0] : self.functions_indices[
                "otlcircuit"
            ][1],
        ]
        X4 = X[
            :, self.functions_indices["piston"][0] : self.functions_indices["piston"][1]
        ]
        Xdummy = X[
            :, self.functions_indices["dummy"][0] : self.functions_indices["dummy"][1]
        ]

        y_ = np.empty((X.shape[0], 4))
        y_[:] = np.nan
        y_[:, 0] = self.borehole(X1)
        y_[:, 1] = self.wingweight(X2)
        y_[:, 2] = self.otlcircuit(X3)
        y_[:, 3] = self.piston(X4)

        miny = np.tile(np.min(y_, axis=1), (4, 1)).T
        maxy = np.tile(np.max(y_, axis=1), (4, 1)).T

        y = (y_ - miny) / (maxy - miny)
        y = np.sum(y, axis=1)
        if self.num_dummy > 0:
            y += (
                np.sum(Xdummy, axis=1) / self.num_dummy / 10
            )  # TODO add dummy differently

        return y


class SobolLevitan(ModelBase):
    """Class that implements the Sobol-Levitan function.

    Parameters
    ----------
    num_params : int
        Number of model inputs
    num_influential : int
        Number of influential inputs
    case : str
        Can take values `easy` and `hard`, where `easy` corresponds to ``b[:num_influential]=1`` and the rest to 0,
        so that influential inputs are clearly active.
        Whereas `hard` corresponds to setting b[:20] to an array of gradually decreasing values, and the rest to 0.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Original paper:
        :cite:ts:`sobol1999use`
    Other paper:
        :cite:ts:`moon2012two`
    Useful link:
        http://www.sfu.ca/~ssurjano/soblev99.html

    """

    def __init__(self, num_params=None, num_influential=None, case="hard"):

        if not num_params:
            num_params = 60
        if not num_influential:
            num_influential = 8
        assert num_influential <= num_params

        if case == "easy":
            b = np.zeros(num_params)
            b[:num_influential] = 1
        elif case == "hard":
            b = np.zeros(num_params)
            b[:20] = np.array(
                [
                    2.6795,
                    2.2289,
                    1.8351,
                    1.4938,
                    1.2004,
                    0.9507,
                    0.7406,
                    0.5659,
                    0.4228,
                    0.3077,
                    0.2169,
                    0.1471,
                    0.0951,
                    0.0577,
                    0.0323,
                    0.0161,
                    0.0068,
                    0.0021,
                    0.0004,
                    0.0,
                ]
            )

        self.influential_params = np.arange(
            num_influential
        )  # we already know for this function, for comparing with GSA results

        # Set b and c0 values as in the Moon paper
        self.num_params = num_params
        self.num_influential = num_influential
        self.influential_params = self.influential_params
        self.case = case
        self.b = b
        self.c0 = 0

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        Id = (np.exp(self.b) - 1) / self.b
        Id[np.isnan(Id)] = 1
        y = np.exp(np.sum(self.b * X, axis=1)) - np.prod(Id) + self.c0
        return y


class SobolG(ModelBase):
    """Class that implements the Sobol G function.

    Parameters
    ----------
    num_params : int
        Number of model inputs
    num_influential : int
        Number of influential inputs
    a : np.array of size [num_params, 1]
        Coefficients for each model input, which determine input importance. Lower ``a`` indicates higher importance.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    References
    ----------
    Paper:
        :cite:ts:`saltelli2010variance`
    Useful link:
        http://www.sfu.ca/~ssurjano/gfunc.html
        https://www.gdr-mascotnum.fr/media/impec07_crestaux.pdf - default ``a`` values

    """

    def __init__(self, num_params=50, num_influential=5, a=None):

        assert num_influential <= num_params
        self.num_params = num_params
        self.num_influential = num_influential
        if a is None:
            a = (np.arange(1, self.num_params + 1) - 2) / 2
        self.a = a
        assert len(self.a) == self.num_params
        self.sensitivity_indices = self.get_sensitivity_indices()
        self.influential_params = np.argsort(self.sensitivity_indices)[
            -self.num_influential :
        ]  # we already know for this function, for comparing with GSA results

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        y = np.prod((np.abs(4 * X - 2) + self.a) / (1 + self.a), axis=1)
        return y

    def get_sensitivity_indices(self):
        """Computes analytical first and total order Sobol indices."""
        V1 = 1 / 3 / (1 + self.a) ** 2
        product = np.tile(np.prod(1 + V1), self.num_params)
        VT = V1 * product / (1 + V1)
        V = product - 1
        first = V1 / V
        total = VT / V
        dict_ = {
            "First order Sobol": first,
            "Total order Sobol": total,
        }
        return dict_


class SobolGstar(ModelBase):
    """Class that implements the Sobol G_star function.

    Setting alpha=1 and delta=0, reverts Sobol G_star into Sobol G function

    Parameters
    ----------
    num_params : int
        Number of model inputs
    num_influential : int
        Number of influential inputs
    a : np.array of size ``num_params``
        Coefficients for each model input, which determine input importance. Lower ``a`` indicates higher importance.
    alpha : np.array of size ``num_params``
        Default value is 1 for all parameters.
    delta : np.array of size ``num_params``
        Default value is 0 for all parameters.

    Returns
    -------
    y : np.array of size ``iterations``
        Model outputs.

    References
    ----------
    Paper:
        :cite:ts:`saltelli2010variance`
    Useful link:
        http://www.sfu.ca/~ssurjano/gfunc.html
        https://www.gdr-mascotnum.fr/media/impec07_crestaux.pdf - default ``a`` values

    """

    def __init__(
        self, num_params=50, num_influential=None, a=None, alpha=None, delta=None
    ):

        assert num_influential <= num_params
        self.num_params = num_params
        if num_influential is None:
            num_influential = int(0.1 * self.num_params)
        self.num_influential = num_influential
        self.influential_params = np.arange(self.num_influential)
        if a is None:
            a = np.hstack(
                [
                    np.ones(self.num_influential) * 0.9,
                    np.ones(self.num_params - self.num_influential) * 9,
                ]
            )
            # a = (np.arange(1, self.num_params + 1) - 2) / 2
        self.a = a
        if alpha is None:
            alpha = np.ones(self.num_params)
        self.alpha = alpha
        if delta is None:
            delta = np.zeros(self.num_params)
        self.delta = delta
        assert len(self.a) == len(self.alpha) == len(self.delta) == self.num_params
        self.S_dict_analytical = self.get_sensitivity_indices()
        self.S_boolean = self.get_boolean_indices()

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        y = np.prod(
            (
                (1 + self.alpha)
                * np.abs(2 * (X + self.delta - np.floor(X + self.delta)) - 1)
                ** self.alpha
                + self.a
            )
            / (1 + self.a),
            axis=1,
        )
        return y

    def get_sensitivity_indices(self):
        """Computes analytical first and total order Sobol indices."""
        V1 = self.alpha ** 2 / (1 + 2 * self.alpha) / (1 + self.a) ** 2
        product = np.tile(np.prod(1 + V1), self.num_params)
        VT = V1 * product / (1 + V1)
        V = product - 1
        first = V1 / V
        total = VT / V
        S_dict = {
            "First order": first,
            "Total order": total,
        }
        return S_dict

    def get_boolean_indices(self):
        """Returns boolean array with ``True`` values for known influential inputs, and ``False`` - for non-influential."""
        S_boolean = np.hstack(
            [
                np.ones(self.num_influential),
                np.zeros(self.num_params - self.num_influential),
            ]
        )  # we already know for this function, for comparing with GSA results
        return S_boolean


class Nonlinear(ModelBase):
    """Class that implements nonlinear function y = x0(x1-x2).

    Parameters
    ----------
    num_params : int
        Number of model inputs.
    num_influential : int
        Number of influential inputs.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Model outputs.

    """

    def __init__(self):
        self.num_params = 10

    def __len__(self):
        return self.num_params

    def rescale(self, X):
        return X

    def __call__(self, X):
        y = np.zeros(X.shape[0])
        y[:] = X[:, 0] * (X[:, 1] - X[:, 2])
        return y
