## initialisation for data loading module

import sys

sys.path.insert(1, "load_data/")


import importlib

import config
import extract_cmip6
import numpy as np

importlib.reload(extract_cmip6)
importlib.reload(config)

from config import get_config


def get_cmip6_data(models, as_xarray=True, **kwargs):

    config = get_config()

    lon_pc, lat_pc, wgt, wgt_l, idx_l = extract_cmip6.get_land_mask(
        config["dir_meta_data"]
    )

    data = {}

    for model in models:

        data[model] = extract_cmip6.load_data_single_mod(
            config["dir_cmip6_data"], model, idx_l, wgt, as_xarray=as_xarray, **kwargs
        )

    return data


def get_meta_data():

    """
    Returns (in order)

    -lon_pc: centred longitude points
    -lat_pc: centred latitude points
    -wgt: weights for all Earth's surface
    -wgt_l: weights for only land
    -idx_l: land sea mask, True for land

    """

    config = get_config()

    return extract_cmip6.get_land_mask(config["dir_meta_data"])


def get_radius_around_gp(gp, idx_l, r=6):

    """
    Returns all grid points within a radius of input grid point, gp (int).
    """

    lons, lats = np.meshgrid(np.arange(-178.75, 180, 2.5), np.arange(-88.75, 90, 2.5))
    lat_grid = lats[idx_l]
    lon_grid = lons[idx_l]

    r = r
    i_x = np.ma.masked_inside(lat_grid, lat_grid[gp] - r, lat_grid[gp] + r)
    i_y = np.ma.masked_inside(lon_grid, lon_grid[gp] - r, lon_grid[gp] + r)
    r_mask = i_x.mask * i_y.mask

    return np.squeeze(np.argwhere(r_mask))
