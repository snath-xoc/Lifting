import argparse
import importlib

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

import load_data
import reconstruct

importlib.reload(load_data)
importlib.reload(reconstruct)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=0)
    parser.add_argument("--models", action="store", type=str, nargs="*")
    parser.add_argument(
        "--scens",
        action="store",
        type=str,
        nargs="*",
        default=["historical", "ssp585", "ssp126", "ssp245"],
    )

    args = parser.parse_args()

    i_mon = args.month
    models = args.models
    scens = args.scens

    lon_pc, lat_pc, wgt, wgt_l, idx_l = load_data.get_meta_data()

    dir_est = [
        load_data.get_config()["dir_output"],
        load_data.get_config()["dir_output_hurs"],
    ]
    dir_out_emu = ["/trial_emulations/", "/trial_emulations/"]

    data_tas = load_data.get_cmip6_data(models)
    data_rh = load_data.get_cmip6_data(models, var="hurs")

    for model in models:

        data_rh[model]["globmean"] = data_tas[model]["globmean"]

    for model in models:

        for scen in scens:

            if scen == "historical":
                time_beg = "1850-0%i" % (i_mon + 1)
                time_end = "2015-0%i" % (i_mon + 1)

            else:
                time_beg = "2015-0%i" % (i_mon + 1)
                time_end = "2101-0%i" % (i_mon + 1)

            Tanglob_temp = (
                data_tas[model]
                .sel({"scenario": scen})["globmean"]
                .mean("run")
                .groupby("time.year")
                .mean()
                .values
            )
            Tanglob_temp = Tanglob_temp.reshape(-1)[~np.isnan(Tanglob_temp.reshape(-1))]

            nr_ts = Tanglob_temp.shape[0]
            GMT = lowess(
                Tanglob_temp,
                np.arange(nr_ts),
                return_sorted=False,
                frac=50 / nr_ts,
                it=0,
            )

            reconstruct.create_emulations(
                dir_est,
                dir_out_emu,
                i_mon,
                model,
                GMT,
                scen,
                var=["tas", "hurs"],
                time_beg=time_beg,
                time_end=time_end,
            )
