## Some functions for fitting used when fitting

import importlib
import os
from collections import defaultdict

import joblib
import numpy as np
import regionmask
import xarray as xr
from helper_weighted_fit import fit_weighted
from joblib import Parallel, delayed
from pygam import ExpectileGAM, LinearGAM, l, s, te
from scipy.stats import linregress, norm
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

import lifting

importlib.reload(lifting)


def get_gam_arch(gam_arch):

    if gam_arch == "linear":

        return LinearGAM(s(0) + l(1))

    if gam_arch == "spline":

        return LinearGAM(s(0) + s(1))

    elif gam_arch == "inter":

        return LinearGAM(te(0, 1, constraints="concave"))

    elif gam_arch == "exp":

        return ExpectileGAM(s(0, constraints="concave"), expectile=0.5)


def fit_mean_response(
    dir_out,
    models,
    scen_train,
    data,
    idx_l,
    wgt_l,
    linreg=False,
    weighted=False,
    gam_arch="spline",
):

    """

    Fitting Linear regression or GAM for local, monthly temperatures using GMT as covariate.
    For linear regression, a grid point- and month- specific model is fitted, for GAM a grid point-
    specific model is fitted.

    Inputs
    ------

    dir_out: string
             Directory for output, see config in load data

    models: list
            list of model names in string format

    scen_train: list
            list of scenarios to use for training in string format

    data_tas: xarray
            xr.DataArray of "y", "globmean", "ref", with dimensions (scen, run, time, gp)

    linreg: Boolean
            True if wanting to fit linear regression and not GAM, default=False

    weighted: Boolean
            True if wanting to weight data in instance of heteroskedasticity, default=False

    gam_arch: string
            one of linear, spline or inter, specifying GAM architecture, default=spline

    Saves model under dir_out+model subfolder alongside residuals
    """

    for model in models:

        if not os.path.exists(dir_out + model + "/mean_response/"):

            os.makedirs(dir_out + model + "/mean_response/")

        Tanglob_train = []

        for scen in scen_train:

            Tanglob_temp = (
                data[model]
                .sel({"scenario": scen, "run": 1})["globmean"]
                .groupby("time.year")
                .mean()
                .values
            )
            Tanglob_temp = Tanglob_temp.reshape(-1)[~np.isnan(Tanglob_temp.reshape(-1))]

            nr_ts = Tanglob_temp.shape[0]
            Tanglob_train.append(
                lowess(
                    Tanglob_temp,
                    np.arange(nr_ts),
                    return_sorted=False,
                    frac=50 / nr_ts,
                    it=0,
                )
            )

        Tanglob_train = np.hstack((Tanglob_train))
        nr_years = Tanglob_train.shape[0]

        if linreg:

            print("Fitting linear model for, ", model, " over scenarios, ", scen_train)

            coeff = np.zeros([2, 12, idx_l.sum()])
            residuals = np.zeros([nr_years, 12, idx_l.sum()])
            for i_mon in range(12):

                for gp in np.arange(idx_l.sum()):

                    y_train = (
                        data[model]
                        .sel({"scenario": scen_train, "run": 1, "gp": gp})["y"]
                        .values.reshape(-1, 12)[:, i_mon]
                    )
                    y_train = y_train[~np.isnan(y_train)]

                    coeff[0, i_mon, gp], coeff[1, i_mon, gp], _, _, _ = linregress(
                        Tanglob_train, y_train
                    )

                    residuals[:, i_mon, gp] = comp_residuals(
                        coeff[:, i_mon, gp], Tanglob_train, y_train
                    )

            print("Storing files")
            joblib.dump(coeff, dir_out + model + "/mean_response/lm.pkl")
            joblib.dump(residuals, dir_out + model + "/mean_response/lm_residuals.pkl")

        else:

            X_train = Tanglob_train.reshape(1, -1)

            X_lift = data[model].sel({"scenario": scen_train, "run": 1})["y"].values

            lifted = []

            for i_mon in range(12):

                lifted.append(
                    lifting.lift(
                        X_lift,
                        idx_l,
                        i_mon,
                        defined_regions=regionmask.defined_regions.ar6.land,
                    )
                )
                lifted[i_mon].execute_over_regions()

            print("Fitting GAM for, ", model, " over scenarios, ", scen_train)

            ## fit GAM for each grid point in parallel, with thin plate spline over month and GMT
            est = Parallel(n_jobs=10)(
                delayed(fit_monthly)(
                    gam_arch,
                    X_train.T,
                    np.hstack(
                        (
                            [
                                lifted[i_mon]
                                .C[region][lifted[i_mon].max_iter[region]][:, 0]
                                .reshape(-1, 1)
                                for i_mon in range(12)
                            ]
                        )
                    ),
                    weighted=weighted,
                )
                for region in lifted[0].pairs.keys()
            )

            print("Computing residuals")

            res_temp = Parallel(n_jobs=10)(
                delayed(comp_residuals_gam)(
                    est[i_reg],
                    X_train.T,
                    np.hstack(
                        (
                            [
                                lifted[i_mon]
                                .C[region][lifted[i_mon].max_iter[region]][:, 0]
                                .reshape(-1, 1)
                                for i_mon in range(12)
                            ]
                        )
                    ),
                    monthly=True,
                )
                for i_reg, region in enumerate(lifted[0].pairs.keys())
            )

            residuals = np.hstack(
                [res_temp[i].reshape(-1, 1) for i in range(len(res_temp))]
            )

            residuals_gp = comp_residuals_lifted(est, lifted, X_train.T, X_lift)

            print("Storing files")

            joblib.dump(lifted, dir_out + model + "/mean_response/lifted.pkl")

            if gam_arch == "spline":  ## for keeping conventions

                if weighted:
                    joblib.dump(
                        est, dir_out + model + "/mean_response/gam_reglift_wgt.pkl"
                    )
                    joblib.dump(
                        residuals,
                        dir_out
                        + model
                        + "/mean_response/gam_residuals_reglift_wgt.pkl",
                    )
                else:
                    joblib.dump(est, dir_out + model + "/mean_response/gam_reglift.pkl")
                    joblib.dump(
                        residuals,
                        dir_out + model + "/mean_response/gam_residuals_reglift.pkl",
                    )
                    joblib.dump(
                        residuals_gp,
                        dir_out + model + "/mean_response/residuals_lifted.pkl",
                    )
            else:

                if weighted:
                    joblib.dump(
                        est,
                        dir_out
                        + model
                        + "/mean_response/gam_"
                        + gam_arch
                        + "reglift_wgt.pkl",
                    )
                    joblib.dump(
                        residuals,
                        dir_out
                        + model
                        + "/mean_response/gam_"
                        + gam_arch
                        + "_residuals_reglift_wgt.pkl",
                    )
                    joblib.dump(
                        residuals_gp,
                        dir_out
                        + model
                        + "/mean_response/"
                        + gam_arch
                        + "_residuals_lifted_wgt.pkl",
                    )
                else:
                    joblib.dump(
                        est,
                        dir_out
                        + model
                        + "/mean_response/gam_"
                        + gam_arch
                        + "_reglift.pkl",
                    )
                    joblib.dump(
                        residuals,
                        dir_out
                        + model
                        + "/mean_response/gam_"
                        + gam_arch
                        + "_residuals_reglift.pkl",
                    )
                    joblib.dump(
                        residuals_gp,
                        dir_out
                        + model
                        + "/mean_response/"
                        + gam_arch
                        + "_residuals_lifted.pkl",
                    )

    return


def fit_monthly(gam_arch, X, y, weighted=False):

    ests = []
    y = y.reshape(-1, 12)

    for i_mon in range(12):

        if weighted:
            est = get_gam_arch(gam_arch)
            ests.append(fit_weighted(est, X, y[:, i_mon]))

        else:
            est = get_gam_arch(gam_arch)
            ests.append(est.gridsearch(X, y[:, i_mon], progress=False))

    return ests


def comp_residuals(coeff, X, y):

    """
    Compute residuals between predictions made with model (est: pygam object) using predictors (X: ndarray)
    and actual values (y: ndarray)

    """

    pred = coeff[0] * X + coeff[1]

    return y - pred


def comp_residuals_lifted(est, lifted, X, y):

    """
    Compute residuals between predictions made with model (est: pygam object) using predictors (X: ndarray)
    and actual values (y: ndarray)
    """

    y = y[~np.isnan(y)].reshape(-1, 12, 2652)  ## hardcoded no. land grid points
    res = np.zeros_like(y)

    grounded_grid = Parallel(n_jobs=10)(
        delayed(lifted[i_mon].ground_over_regions)(
            [
                est[i_reg][i_mon].predict(X)
                for i_reg, region in enumerate(lifted[0].pairs.keys())
            ],
            n_samp=1,
            parallel=False,
        )
        for i_mon in range(12)
    )

    for i_mon in range(12):

        res[:, i_mon, :] = y[:, i_mon] - grounded_grid[i_mon]

    return res


def comp_residuals_gam(est, X, y, monthly=False):

    """
    Compute residuals between predictions made with model (est: pygam object) using predictors (X: ndarray)
    and actual values (y: ndarray)
    """

    if monthly:

        y = y.reshape(-1, 12)
        res = np.zeros_like(y)

        for i_mon in range(12):

            pred = est[i_mon].predict(X)
            res[:, i_mon] = y[:, i_mon] - pred

        return res

    else:

        pred = est.predict(X)

        return y - pred
