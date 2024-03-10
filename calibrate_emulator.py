import argparse
import importlib
import sys

import fit_tools
import load_data

importlib.reload(load_data)
importlib.reload(fit_tools)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store", type=str, nargs="*")
    parser.add_argument(
        "--scen_train",
        action="store",
        type=str,
        nargs="*",
        default=["historical", "ssp585", "ssp126"],
    )
    parser.add_argument("--gam_arch", type=str, default="exp")
    args = parser.parse_args()

    models = args.models
    scen_train = args.scen_train
    gam_arch = args.gam_arch

    lon_pc, lat_pc, wgt, wgt_l, idx_l = load_data.get_meta_data()

    data_tas = load_data.get_cmip6_data(models)
    data_rh = load_data.get_cmip6_data(models, var="hurs")

    for model in models:

        data_rh[model]["globmean"] = data_tas[model]["globmean"]

    fit_tools.fit_mean_response(
        load_data.get_config()["dir_output"],
        models,
        scen_train,
        data_tas,
        idx_l,
        wgt_l,
        linreg=False,
        weighted=False,
        gam_arch=gam_arch,
    )

    fit_tools.fit_mean_response(
        load_data.get_config()["dir_output_hurs"],
        models,
        scen_train,
        data_rh,
        idx_l,
        wgt_l,
        linreg=False,
        weighted=False,
        gam_arch=gam_arch,
    )
