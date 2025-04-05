## Helper functions for weighted fit
## We tried to implement dynamical weighting such that for PIRLS a(phi) = phi/w and w are dynamically adjusted
## Helps (or is meant to) account for heteroskedasticity

import numpy as np
from pygam import ExpectileGAM, LinearGAM, l, s, te
from pygam.utils import check_X, check_X_y, check_y
from scipy import sparse
from scipy.optimize import minimize, rosen_der
from scipy.signal import detrend

EPS = np.finfo(np.float64).eps


def rolling_std(t, n):

    l = t.shape[0]
    m = np.zeros(l)

    for i in range(n // 2):
        m[i] = np.std(t[i : i + n])
        m[l - i - 1] = np.std(t[l - i - n - 1 : l - i - 1])

    for i in range(n // 2, l - n // 2):
        m[i] = np.std(t[i - n // 2 : i + n // 2])

    return m


def comp_weights(X, y, weighted=False):

    """
    Deal with heteroskedasticity in predictands (y: ndarray) by computing weights based on changing scale parameter

    References
    ----------

    https://medium.com/datamotus/solving-the-problem-of-heteroscedasticity-through-weighted-regression-e4a22f1afa6b

    !!!DON'T USE ANYMORE DOWNGRADED!!!

    """

    if weighted:

        y = y.reshape(-1, 12)

        nr_ts = X.shape[0]

        idx_sort = np.argsort(X)
        y_sorted = detrend(y)[idx_sort]

        spl = []

        for i_mon in range(12):

            ## compute rolling standard deviation
            roll_std = rolling_std(y_sorted[:, i_mon], 50)
            spl.append(np.poly1d(np.polyfit(X[idx_sort], roll_std, 3)))
            # spl.append(lowess(roll_std,X[idx_sort],return_sorted=False,frac=50/nr_ts,it=0))

        return np.vstack(([spl[i_mon](X) for i_mon in range(12)])).flatten()

    else:

        return None


def _pirls_weighted(self, X, Y):

    """
    Performs stable PIRLS iterations to estimate GAM coefficients, whilst dynamically calculating and adjusting weights
    to account for heteroskedasticity

    Parameters
    ---------
    X : array-like of shape (n_samples, m_features)
        containing input data
    Y : array-like of shape (n,)
        containing target data

    Returns
    -------
    None
    """
    modelmat = self._modelmat(X)  # build a basis matrix for the GLM
    n, m = modelmat.shape

    # initialize GLM coefficients if model is not yet fitted
    if (
        not self._is_fitted
        or len(self.coef_) != self.terms.n_coefs
        or not np.isfinite(self.coef_).all()
    ):
        # initialize the model
        self.coef_ = self._initial_estimate(Y, modelmat)

    assert np.isfinite(
        self.coef_
    ).all(), "coefficients should be well-behaved, but found: {}".format(self.coef_)

    P = self._P()
    S = sparse.diags(np.ones(m) * np.sqrt(EPS))  # improve condition
    # S += self._H # add any user-chosen minumum penalty to the diagonal

    # if we dont have any constraints, then do cholesky now
    if not self.terms.hasconstraint:
        E = self._cholesky(S + P, sparse=False, verbose=self.verbose)

    min_n_m = np.min([m, n])
    Dinv = np.zeros((min_n_m + m, m)).T

    for _ in range(self.max_iter):
        # recompute cholesky if needed
        if self.terms.hasconstraint:
            P = self._P()
            C = self._C()
            E = self._cholesky(S + P + C, sparse=False, verbose=self.verbose)

        # forward pass
        y = Y.copy()  # for simplicity
        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)

        # Compute month-specific weights dynamically

        def _weights_func(args):

            """
            args: coefficients for linear regression

            """

            a, b, c = args

            weights = a * (self.cov**2) + b * (self.cov) + c

            return self.residuals / weights

        def _neglogpdf(args):

            """
            Compute loglikelihood with repsect to normal dist with loc=0, scale=1
            Used to calculate weighting function since we want residuals to be as Gaussian as possible
            """

            res_norm = _weights_func(args)

            return -(norm.logpdf(res_norm).sum())

        self.m = {}
        self._weights = np.zeros_like(y).reshape(-1, 12)

        for i_mon in range(12):

            self.cov = X[:, 1].reshape(-1, 12)[:, i_mon]
            self.residuals = (y - mu).reshape(-1, 12)[:, i_mon]

            self.m[i_mon] = minimize(
                _neglogpdf,
                x0=[1, 1, 1],
                method="SLSQP",
            )

            if self.m[i_mon].success:

                self.m[i_mon] = self.m[i_mon].x
                self._weights[:, i_mon] = (
                    self.m[i_mon][0] * self.cov**2
                    + self.m[i_mon][1] * self.cov
                    + self.m[i_mon][2]
                )

            else:

                self.m[i_mon] = None
                self._weights[:, i_mon] = np.ones_like(self.cov)

        W = self._W(mu, self._weights.flatten(), y)  # create pirls weight matrix

        # check for weghts == 0, nan, and update
        mask = self._mask(W.diagonal())
        y = y[mask]  # update
        lp = lp[mask]  # update
        mu = mu[mask]  # update
        W = sparse.diags(W.diagonal()[mask])  # update

        # PIRLS Wood pg 183
        pseudo_data = W.dot(self._pseudo_data(y, lp, mu))

        # log on-loop-start stats
        self._on_loop_start(vars())

        WB = W.dot(modelmat[mask, :])  # common matrix product
        Q, R = np.linalg.qr(WB.A)

        if not np.isfinite(Q).all() or not np.isfinite(R).all():
            raise ValueError("QR decomposition produced NaN or Inf. " "Check X data.")

        # need to recompute the number of singular values
        min_n_m = np.min([m, n, mask.sum()])
        Dinv = np.zeros((m, min_n_m))

        # SVD
        U, d, Vt = np.linalg.svd(np.vstack([R, E]))

        # mask out small singular values
        # svd_mask = d <= (d.max() * np.sqrt(EPS))

        np.fill_diagonal(Dinv, d**-1)  # invert the singular values
        U1 = U[:min_n_m, :min_n_m]  # keep only top corner of U

        # update coefficients
        B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
        coef_new = B.dot(pseudo_data).flatten()
        diff = np.linalg.norm(self.coef_ - coef_new) / np.linalg.norm(coef_new)
        self.coef_ = coef_new  # update

        # log on-loop-end stats
        self._on_loop_end(vars())

        # check convergence
        if diff < self.tol:
            break

    # estimate statistics even if not converged
    self._estimate_model_statistics(
        Y, modelmat, inner=None, BW=WB.T, B=B, weights=self._weights.flatten(), U1=U1
    )

    if diff < self.tol:
        return

    print("did not converge")
    return


def fit_weighted(self, X, y):

    # validate parameters
    self._validate_params()

    # validate data
    y = check_y(y, self.link, self.distribution, verbose=self.verbose)
    X = check_X(X, verbose=self.verbose)
    check_X_y(X, y)

    # validate data-dependent parameters
    self._validate_data_dep_params(X)

    # set up logging
    if not hasattr(self, "logs_"):
        self.logs_ = defaultdict(list)

    # begin capturing statistics
    self.statistics_ = {}
    self.statistics_["n_samples"] = len(y)
    self.statistics_["m_features"] = X.shape[1]

    # optimize
    _pirls_weighted(self, X, y)

    return self
