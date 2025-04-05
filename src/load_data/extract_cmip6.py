## Load cmip6 data from a given directory (assuming naming and structural conventions follow new generation)

import glob

import mplotutils as mpu
import numpy as np
import regionmask
import xarray as xr


def mask_percentage(regions, lon, lat, **kwargs):

    """
    Sample with 10 times higher resolution.

    Notes
    -----
    - assumes equally-spaced lat & lon!
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38 in
      August 2020
    - prototype of what will eventually be integrated in his regionmask package

    """

    lon_sampled = sample_coord(lon)
    lat_sampled = sample_coord(lat)

    mask = regions.mask(lon_sampled, lat_sampled, **kwargs)

    isnan = np.isnan(mask.values)

    numbers = np.unique(mask.values[~isnan])
    numbers = numbers.astype(int)

    mask_sampled = list()
    for num in numbers:
        # coarsen the mask again
        mask_coarse = (mask == num).coarsen(lat=10, lon=10).mean()
        mask_sampled.append(mask_coarse)

    mask_sampled = xr.concat(
        mask_sampled, dim="region", compat="override", coords="minimal"
    )

    mask_sampled = mask_sampled.assign_coords(region=("region", numbers))

    return mask_sampled


def sample_coord(coord):

    """

    Sample coords for the percentage overlap.

    Notes
    -----
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38
      in August 2020
    -> prototype of what will eventually be integrated in his regionmask package

    """
    d_coord = coord[1] - coord[0]

    n_cells = len(coord)

    left = coord[0] - d_coord / 2 + d_coord / 20
    right = coord[-1] + d_coord / 2 - d_coord / 20

    return np.linspace(left, right, n_cells * 10)


def get_land_mask(dir_data):

    """
    Returns
    -------

    -lon_pc: centred longitude points
    -lat_pc: centred latitude points
    -wgt: weights for all Earth's surface
    -wgt_l: weights for only land
    -idx_l: land sea mask, True for land

    """

    frac_l = xr.open_mfdataset(
        dir_data + "interim_invariant_lsmask_regrid.nc",
        combine="by_coords",
        decode_times=False,
    )

    # land-sea mask of ERA-interim bilinearily interpolated
    frac_l_raw = np.squeeze(frac_l.lsm.values.copy())
    frac_l = frac_l.where(
        frac_l.lat > -60, 0
    )  # remove Antarctica from frac_l field (ie set frac l to 0)
    idx_l = (
        np.squeeze(frac_l.lsm.values) > 0.43
    )  # idex land #-> everything >0 I consider land

    lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat)
    lons, lats = np.meshgrid(frac_l.lon, frac_l.lat)

    # obtain a (subsampled) land-sea mask
    ls = {}
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    # gives fraction of land -> in extract_land() script decide above which land
    # fraction threshold to consider a grid point as a land grid point
    ls["grid_raw"] = np.squeeze(
        mask_percentage(land_110, frac_l.lon, frac_l.lat).values
    )

    # remove Antarctica
    idx_ANT = np.where(frac_l.lat < -60)[0]
    ls["grid_no_ANT"] = ls["grid_raw"].copy()
    ls["grid_no_ANT"][idx_ANT] = 0

    idx_l = ls["grid_no_ANT"] > 1 / 3

    wgt = np.cos(np.deg2rad(lats))  # area weights of each grid point
    wgt_l = (wgt * frac_l_raw)[
        idx_l
    ]  # area weights for land grid points (including taking fraction land into consideration)

    return lon_pc, lat_pc, wgt, wgt_l, idx_l


def as_xr(
    y,
    run_nrs,
    T_ref,
    Tanglob_wgt,
    time_start="1850-01",
    time_stop="2101-01",
    usr_time_res="mon",
):

    """
    Inputs:
    ------

    y: Nested dictionary with scenario and then run number, contains ndarray (time, gp)
       land values for a given variable

    run_nrs: Nested dictionary with scenario
       number of runs

    T_ref: Nested dictionary with scenario and then run number, contains ndarray (time, gp)
       reference values for variable over globe (note only land)

    Tanglob_wgt_mean: Nested dictionary with scenario and then run number, contains ndarray (time,)
       globally weighted alue of variable (includes oceans)

    Returns:
    --------

    xarray of dimensions (scenario, run_nr, time, gp)

    """

    xr_concat = []

    for scen in y.keys():

        if scen == "historical":

            time_beg = time_start
            time_end = "2015-01"
        else:

            time_beg = "2015-01"
            time_end = time_stop

        if usr_time_res == "mon":

            times = np.arange(
                time_beg, time_end, np.timedelta64(1, "M"), dtype="datetime64"
            )

        elif usr_time_res == "ann":

            times = np.arange(
                time_beg, time_end, np.timedelta64(1, "Y"), dtype="datetime64"
            )

        else:

            print("Time resolution not recognised/implemented")

            return

        y_vals = np.stack(([y[scen][run] for run in run_nrs[scen]]))
        Tanglob_wgt_vals = np.stack(([Tanglob_wgt[scen][run] for run in run_nrs[scen]]))

        xr_concat.append(
            xr.Dataset(
                {
                    "y": (["scenario", "run", "time", "gp"], y_vals[None, :]),
                    "ref": (["gp"], T_ref),
                    "globmean": (
                        ["scenario", "run", "time"],
                        Tanglob_wgt_vals[None, :],
                    ),
                },
                coords={
                    "scenario": (["scenario"], [scen]),
                    "run": (["run"], np.asarray(run_nrs[scen])),
                    "time": (["time"], times),
                    "gp": (["gp"], np.arange(y_vals.shape[2])),
                },
            )
        )

    return xr.concat(xr_concat, dim="scenario")


def load_data_single_mod(
    dir_data,
    model,
    idx_l,
    wgt,
    Tref_start="1850-01-01",
    Tref_end="1900-01-01",
    usr_time_res="mon",
    var="tas",
    as_xarray=True,
    **kwargs
):

    """Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

    Keyword argument:
    - model: model str
    - scenario: scenario str
    - Tanglob_idx: decides if wgt Tanglob is computed (and returned) or not, default is not returned
    - Tref_all: decides if the Tref at each grid point is dervied based on all available runs or not, default is yes
    - Tref_start: starting point for the reference period with default 1870
    - Tref_end: first year to no longer be included in reference period with default 1900

    Output:
    - y: dictionary the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the
    srexgrid) for each scenario available
    - run_nrs: dictionary of run numbers available within each scenario
    - Tref: reference variable values (in case absolute values needed)
    - Tan_wgt_globmean = area weighted global mean variable

    """
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
    # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).

    # right now I keep reloading constants fields for each run I add -> does not really make sense.
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
    print("start with model", model)

    # vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)
    temp_res = usr_time_res  # if not, reading the var file needs to be changed as time var is not named in the same way anymore
    spatial_res = "g025"

    Tan_wgt_globmean = {}
    y = {}
    T_ref = np.zeros(idx_l.shape)
    run_nrs = {}

    dir_var = dir_data + var + "/" + usr_time_res + "/" + spatial_res + "/"

    if var == "hurs":

        dir_var = (
            dir_data + var + "/" + var + "/" + usr_time_res + "/" + spatial_res + "/"
        )

    run_names_list = sorted(
        glob.glob(
            dir_var
            + var
            + "_"
            + temp_res
            + "_"
            + model
            + "_ssp*_"
            + "r*i1p1f*"
            + "_"
            + spatial_res
            + ".nc"
        )
    )
    run_names_list_historical = sorted(
        glob.glob(
            dir_var
            + var
            + "_"
            + temp_res
            + "_"
            + model
            + "_historical_"
            + "r*i1p1f*"
            + "_"
            + spatial_res
            + ".nc"
        )
    )

    for run_name in run_names_list:
        run_name_ssp = run_name
        data = (
            xr.open_mfdataset(run_name_ssp)
            .sel(time=slice("1850-01-01", "2101-01-01"))
            .roll(lon=72)
        )
        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so same labels as others
        scen = run_name.split("/")[-1].split("_")[-3]
        run = int(run_name.split("/")[-1].split("_")[-2].split("r")[1].split("i")[0])

        if "-over" in run_name:
            continue
        elif "p4" in run_name:
            continue

        if scen not in list(y.keys()):

            y[scen] = {}
            Tan_wgt_globmean[scen] = {}
            run_nrs[scen] = []

            y[scen][run] = data[
                var
            ].values  # still absolute values + still contains sea pixels
            Tan_wgt_globmean[scen][run] = (
                np.multiply(y[scen][run], wgt[None, :, :]).sum(axis=(1, 2)) / wgt.sum()
            )  # area weighted but abs value
            run_nrs[scen].append(run)

        else:
            y[scen][run] = data[
                var
            ].values  # still absolute values + still contains sea pixels
            Tan_wgt_globmean[scen][run] = (
                np.multiply(y[scen][run], wgt[None, :, :]).sum(axis=(1, 2)) / wgt.sum()
            )  # area weighted but abs value
            run_nrs[scen].append(run)

    y["historical"] = {}
    Tan_wgt_globmean["historical"] = {}
    run_nrs["historical"] = []

    for run_name in run_names_list_historical:

        run_name_hist = run_name
        data = (
            xr.open_mfdataset(run_name_hist)
            .sel(time=slice("1850-01-01", "2101-01-01"))
            .roll(lon=72)
        )

        data = data.assign_coords(
            lon=(((data.lon + 180) % 360) - 180)
        )  # assign_coords so same labels as others
        run = int(run_name.split("/")[-1].split("_")[-2].split("r")[1].split("i")[0])

        y["historical"][run] = data[
            var
        ].values  # still absolute values + still contains sea pixels
        Tan_wgt_globmean["historical"][run] = (
            np.multiply(y["historical"][run], wgt[None, :, :]).sum(axis=(1, 2))
            / wgt.sum()
        )
        run_nrs["historical"].append(run)

        T_ref += (
            data[var].sel(time=slice(Tref_start, Tref_end)).mean(dim="time").values
            * 1.0
            / len(run_names_list_historical)
        )  # sum up all ref climates

    T_ref_glob = np.average(T_ref, weights=wgt, axis=(0, 1))

    # obtain the anomalies
    for scen in [i for i in y.keys()]:

        for run in run_nrs[scen]:

            try:
                y[scen][run] = (y[scen][run] - T_ref)[:, idx_l]
                Tan_wgt_globmean[scen][run] = Tan_wgt_globmean[scen][run] - T_ref_glob
            except:
                print(scen, y[scen][run].shape, T_ref.shape)

    if as_xarray:

        kwargs_as_xarray = ["time_start", "time_stop"]
        kwargs_as_xarray_dict = {
            k: kwargs.pop(k) for k in dict(kwargs) if k in kwargs_as_xarray
        }

        return as_xr(
            y,
            run_nrs,
            T_ref[idx_l],
            Tan_wgt_globmean,
            usr_time_res=usr_time_res,
            **kwargs_as_xarray_dict
        )

    else:
        # print('returning all values')
        return y, run_nrs, T_ref, Tan_wgt_globmean
