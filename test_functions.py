import numpy as np
# import brightway2 as bw
# from setup_files import setup_bw_project_simple


# Utils functions
def uniform_rescale(X, inputs):
    left_rescale  = np.array(list(inputs.values()))[:, 0]
    right_rescale = np.array(list(inputs.values()))[:, 1]
    X_rescaled = (right_rescale - left_rescale) * X + left_rescale
    return X_rescaled


# Test functions
class Morris:
    '''
    Morris function
    ---------------
    Source:
        Sampling plans based on balanced incomplete block designs for evaluating the importance of computer model inputs
        Max D. Morris, Leslie M. Moore, Michael D.McKay, 2006
        https://doi.org/10.1016/j.jspi.2005.01.001
    Links:
        http://www.sfu.ca/~ssurjano/morretal06.html (there is a typo in the formula, trust the paper)

    Parameters
    ----------
    num_params : int
        Number of model inputs
    num_influential : int
        Number of influential inputs

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''

    def __init__(self, num_params=None, num_influential=None):

        if not num_params:
            num_params = 10
        if not num_influential:
            num_influential = int(0.1 * num_params) # if n_inf not defined, set 10% of inputs to be influential
        assert num_influential <= num_params

        self.influential_params = np.arange(num_influential) # we already know for this function, for comparing with GSA results

        alpha = np.sqrt(12) - 6*np.sqrt(0.1 * (num_influential - 1))
        beta  = 12 * np.sqrt(0.1 * (num_influential - 1))

        self.num_params = num_params
        self.num_influential = num_influential
        self.influential_params = self.influential_params
        self.alpha = alpha
        self.beta = beta

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return X

    def __call__(self, X):
        y = np.zeros(X.shape[0])
        y[:] = self.alpha*np.sum(X, axis=1)
        for i in range(self.num_influential):
            for j in range(i+1, self.num_influential):
                y[:] += self.beta * X[:,i] * X[:,j]
        return y


class Borehole:
    '''
    Borehole function
    ---------------
    Original source:
        Sensitivity/uncertainty analysis of a borehole scenario comparing
        Latin Hypercube Sampling and deterministic sensitivity approaches.
        Harper, W. V., Gupta, S. K., 1983
        - here the function is slightly different than in the below paper
    Source:
        Two-stage sensitivity-based group screening in computer experiments.
        Moon, H., Dean, A. M., & Santner, T. J., 2012
        https://doi.org/10.1080/00401706.2012.725994
    Links:
        http://www.sfu.ca/~ssurjano/borehole.html

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''
    def __init__(self):
        self.params = {
            'rw': [0.05, 0.15],     # radius of borehole (m)
            'r':  [100, 50000],     # radius of influence (m)
            'Tu': [63070, 115600],  # transmissivity of upper aquifer (m2/yr)
            'Hu': [990, 1110],      # potentiometric head of upper aquifer (m)
            'Tl': [63.1, 116],      # transmissivity of lower aquifer (m2/yr)
            'Hl': [700, 820],       # potentiometric head of lower aquifer (m)
            'L':  [1120, 1680],     # length of borehole (m)
            'Kw': [9855, 12045],    # hydraulic conductivity of borehole (m/yr)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array([3,5]) # TODO check correcteness in the literature

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):
        rw = X[:,0]
        r  = X[:,1]
        Tu = X[:,2]
        Hu = X[:,3]
        Tl = X[:,4]
        Hl = X[:,5]
        L  = X[:,6]
        Kw = X[:,7]
        # Response is water flow rate (m3/yr)
        y = 2*np.pi*Tu*(Hu-Hl) \
            / ( np.log(r/rw) * (1 + 2*L*Tu / ( np.log(r/rw)*rw**2*Kw ) + Tu/Tl ) )
        return y


class Wingweight:
    '''
    Wing weight function
    ---------------
    Source:
        Engineering Design via Surrogate Modelling.
        Forrester, 2008
        https://doi.org/10.1002/9780470770801
    Links:
        http://www.sfu.ca/~ssurjano/wingweight.html

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''
    def __init__(self):
        self.params = {
            'Sw':  [150, 200],       # wing area (ft2)
            'Wfw': [220, 300],       # weight of fuel in the wing (lb)
            'A':   [6, 10],          # aspect ratio
            'Lam': [-10, 10],        # quarter-chord sweep (degrees)
            'q':   [16, 45],         # dynamic pressure at cruise (lb/ft2)
            'lam': [0.5, 1],         # taper ratio
            'tc':  [0.08, 0.18],     # aerofoil thickness to chord ratio
            'Nz':  [2.5, 6],         # ultimate load factor
            'Wdg': [1700, 2500],     # flight design gross weight (lb)
            'Wp':  [0.025, 0.08],    # paint weight (lb / ft2)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array([0,9]) # TODO check correcteness in the literature

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):

        Sw =  X[:, 0]
        Wfw = X[:, 1]
        A =   X[:, 2]
        Lam = X[:, 3] / 180 * np.pi  # convert to radian
        q =   X[:, 4]
        lam = X[:, 5]
        tc =  X[:, 6]
        Nz =  X[:, 7]
        Wdg = X[:, 8]
        Wp =  X[:, 9]

        y = 0.036 * Sw**0.758 * Wfw**0.0035 \
            * (A/np.cos(Lam)**2)**0.6 \
            * q**0.006 * lam**0.04 \
            * (100*tc/np.cos(Lam))**(-0.3) \
            * (Nz*Wdg)**0.49 + Sw*Wp

        return y


class OTLcircuit:
    '''
    OTL circuit function
    ---------------
    Source:
        Modeling Data from Computer Experiments: An Empirical Comparison of Kriging with
        MARS and Projection Pursuit Regression
        Einat Neumann Ben-Ari, David M. Steinberg, 2008
        https://doi.org/10.1080/08982110701580930
    Links:
        http://www.sfu.ca/~ssurjano/otlcircuit.html

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''
    def __init__(self):
        self.params = {
            'Rb1':  [50, 150],      # resistance b1 (K-Ohms)
            'Rb2':  [25, 70],       # resistance b2 (K-Ohms)
            'Rf':   [0.5, 3],       # resistance f (K-Ohms)
            'Rc1':  [1.2, 2.5],     # resistance c1 (K-Ohms)
            'Rc2':  [0.25, 1.2],    # resistance c2 (K-Ohms)
            'beta': [50, 300],      # current gain (Amperes)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array([0,1,2,3])  # TODO check correcteness in the literature

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):

        Rb1  = X[:,0]
        Rb2  = X[:,1]
        Rf   = X[:,2]
        Rc1  = X[:,3]
        Rc2  = X[:,4]
        beta = X[:,5]

        Vb1  = 12*Rb2 / (Rb1 + Rb2)
        temp = beta*(Rc2+9) + Rf

        y = (Vb1+0.74)*beta*(Rc2+9) / temp \
            + 11.35*Rf / temp \
            + 0.74*Rf*beta*(Rc2+9) / temp / Rc1

        return y


class Piston:
    '''
    Piston simulation function function
    ---------------
    Source:
        Modeling Data from Computer Experiments: An Empirical Comparison of Kriging with
        MARS and Projection Pursuit Regression
        Einat Neumann Ben-Ari, David M. Steinberg, 2008
        https://doi.org/10.1080/08982110701580930
    Links:
        http://www.sfu.ca/~ssurjano/piston.html

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''
    def __init__(self):
        self.params = {
            'M': [30, 60],  # piston weight (kg)
            'S': [0.005, 0.020], # piston surface area (m2)
            'V0': [0.002, 0.010], # initial gas volume (m3)
            'k': [1000, 5000], # spring coefficient (N/m)
            'P0': [90000, 110000], # atmospheric pressure (N/m2)
            'Ta': [290, 296], # ambient temperature (K)
            'T0': [340, 360], # filling gas temperature (K)
        }
        self.num_params = len(self.params)
        self.influential_params = np.array([0,2,4,5,6])  # TODO check correcteness in the literature

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return uniform_rescale(X, self.params)

    def __call__(self, X):

        M  = X[:,0]
        S  = X[:,1]
        V0 = X[:,2]
        k  = X[:,3]
        P0 = X[:,4]
        Ta = X[:,5]
        T0 = X[:,6]

        A = P0*S + 19.62*M - k*V0/S
        V = S/2/k * ( np.sqrt(A**2 + 4*k*P0*V0/T0*Ta) - A )

        y = 2*np.pi * np.sqrt( M / (k + S**2*P0*V0/T0*Ta/V**2) )

        return y


class Moon:
    '''
    Moon function
    ---------------
    Source:
        Two-stage sensitivity-based group screening in computer experiments.
        Moon, H., Dean, A. M., & Santner, T. J., 2012
        https://doi.org/10.1080/00401706.2012.725994
    Links:
        http://www.sfu.ca/~ssurjano/moonetal12.html

    Parameters
    ----------
    num_dummy : int
        Number of non-influential dummy variables to add to the model

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''
    def __init__(self, num_dummy=29):
        self.functions_indices = dict( # left included, right excluded
            borehole = [0,8],
            wingweight = [8,18],
            otlcircuit = [18,24],
            piston = [24,31],
            dummy = [31,31+num_dummy],
        )
        self.borehole, self.wingweight, self.otlcircuit, self.piston = Borehole(), Wingweight(), OTLcircuit(), Piston()
        self.params = {
            **self.borehole.params,
            **self.wingweight.params,
            **self.otlcircuit.params,
            **self.piston.params,
        }
        self.num_dummy = num_dummy
        if self.num_dummy:
            self.params.update({str(k): [0,1] for k in range(num_dummy)})
        self.num_params = len(self.params)
        self.influential_params = np.hstack([
            self.borehole.influential_params + self.functions_indices['borehole'][0],
            self.wingweight.influential_params + self.functions_indices['wingweight'][0],
            self.otlcircuit.influential_params + self.functions_indices['otlcircuit'][0],
            self.piston.influential_params + self.functions_indices['piston'][0],
        ])
        self.influential_params.sort()

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        X1 = X[:, self.functions_indices['borehole'][0]: self.functions_indices['borehole'][1]]
        X2 = X[:, self.functions_indices['wingweight'][0]: self.functions_indices['wingweight'][1]]
        X3 = X[:, self.functions_indices['otlcircuit'][0]: self.functions_indices['otlcircuit'][1]]
        X4 = X[:, self.functions_indices['piston'][0]: self.functions_indices['piston'][1]]
        Xdummy = X[:, self.functions_indices['dummy'][0]: self.functions_indices['dummy'][1]]
        X1_rescaled = self.borehole.__rescale__(X1)
        X2_rescaled = self.wingweight.__rescale__(X2)
        X3_rescaled = self.otlcircuit.__rescale__(X3)
        X4_rescaled = self.piston.__rescale__(X4)
        X = np.hstack([X1_rescaled, X2_rescaled, X3_rescaled, X4_rescaled, Xdummy])
        return X

    def __call__(self, X):

        X1 = X[ : , self.functions_indices['borehole'][0]   : self.functions_indices['borehole'][1]   ]
        X2 = X[ : , self.functions_indices['wingweight'][0] : self.functions_indices['wingweight'][1] ]
        X3 = X[ : , self.functions_indices['otlcircuit'][0] : self.functions_indices['otlcircuit'][1] ]
        X4 = X[ : , self.functions_indices['piston'][0]     : self.functions_indices['piston'][1]     ]
        Xdummy = X[:, self.functions_indices['dummy'][0]: self.functions_indices['dummy'][1]]

        y_ = np.empty((X.shape[0],4))
        y_[:] = np.nan
        y_[:,0] = self.borehole(X1)
        y_[:,1] = self.wingweight(X2)
        y_[:,2] = self.otlcircuit(X3)
        y_[:,3] = self.piston(X4)

        miny = np.tile(np.min(y_, axis=1), (4, 1)).T
        maxy = np.tile(np.max(y_, axis=1), (4, 1)).T

        y = (y_-miny) / (maxy-miny)
        y = np.sum(y, axis=1)
        y += np.sum(Xdummy, axis=1) / self.num_dummy / 10 #TODO add dummy differently

        return y


class SobolLevitan:
    '''
    Sobol-Levitan function, where coefficients stored in the `b` vector define variables' importance.
    ---------------
    Source:
        Two-stage sensitivity-based group screening in computer experiments.
        Moon, H., Dean, A. M., & Santner, T. J., 2012
        https://doi.org/10.1080/00401706.2012.725994
    Original paper:
        On the use of variance reducing multipliers in Monte Carlo computations of a global sensitivity index
        Sobol' I., Levitan Yu.
        https://doi.org/10.1016/S0010-4655(98)00156-8
    Links:
        http://www.sfu.ca/~ssurjano/soblev99.html

    Parameters
    ----------
    num_params : int
        Number of model inputs
    num_influential : int
        Number of influential inputs
    case : str
        Can take values `easy` and `hard`, where `easy` corresponds to `b[:num_influential]=1` and the rest to 0,
        so that influential inputs are clearly active.
        Whereas `hard` corresponds to setting b[:20] to an array of gradually decreasing values, and the rest to 0.

    Returns
    -------
    y : np.array of size n_samples x 1
        Function output for each input sample
    '''

    def __init__(self, num_params=None, num_influential=None, case='hard'):

        if not num_params:
            num_params = 60
        if not num_influential:
            num_influential = 8
        assert num_influential <= num_params

        if case == 'easy':
            b = np.zeros(num_params)
            b[:num_influential] = 1
        elif case == 'hard':
            b = np.zeros(num_params)
            b[:20] = np.array([2.6795, 2.2289, 1.8351, 1.4938, 1.2004, 0.9507, 0.7406, 0.5659, 0.4228, 0.3077,
                               0.2169, 0.1471, 0.0951, 0.0577, 0.0323, 0.0161, 0.0068, 0.0021, 0.0004, 0.0])

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

    def __num_input_params__(self):
        return self.num_params

    def __rescale__(self, X):
        return X

    def __call__(self, X):
        Id = (np.exp(self.b) - 1)/self.b
        Id[np.isnan(Id)] = 1
        y = np.exp(np.sum(self.b*X, axis=1)) - np.prod(Id) + self.c0
        return y