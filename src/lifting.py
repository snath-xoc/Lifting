## wavelet lifting scheme, see references:
## https://cm-bell-labs.github.io/who/wim/papers/iciam95.pdf
## https://doi.org/10.1016/j.wace.2023.100580

import time

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import regionmask
from joblib import Parallel, delayed
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# from sksparse.cholmod import cholesky as sp_chol


class lift(object):

    """

    Class object for lifting protocol to be applied to a region so as to capture spatial features
    whilst conserving the mean (so far only first order moment is focussed on, however for our use
    in pattern scaling this makes sense)

    Steps:
    -----

    1) Split: split into pairs, and keep track of odd numbered group
    2) Predict: identify x and it's ys within a cluster, calculate the naive regression error
       i.e., wavelet coefficients (d) and the corresponding aggregation coefficient (lam)
    3) Update: Calculate spatial average, c i.e., scaling coefficients (weighted of course using np.cos(np.deg2rad(lat))
       , as x+lambda*d) in the trivial case, lam = 0.5 and d = y-x

    Outputs
    -------

    Nested list of results from for each split, predict and adapt step
    1) clusters: obtained by k-means at each split step
    1) lam: geometry coefficients
    2) d: wavelet coefficients
    3) c: scaling coefficients

    Attributes
    ----------

    fit: requires X ndarray (time, grid points), and n_steps
    transform: once fitted takes X any given X ndarray (time) and downscales to original ndarray (time, grid points)

    Note
    ----

    Split, predict and update steps are repeated until only x exists
    For singular clusters, c = x, and d = 0

    """

    def __init__(
        self,
        X,
        idx_l,
        i_mon,
        compression_factor=0.05,
        defined_regions=regionmask.defined_regions.giorgi,
        grid_resolution=2.5,
    ):

        assert isinstance(X, np.ndarray)
        X = X[~np.isnan(X)].reshape(-1, 12, idx_l.sum())[:, i_mon, :]

        self.idx_l = idx_l
        self.X = X.copy()

        self.grid_resolution = grid_resolution
        lon = np.arange(-178.75, 180, grid_resolution)
        lat = np.arange(-88.75, 90, grid_resolution)

        lons, lats = np.meshgrid(lon, lat)
        self.lons = lons
        self.lats = lats

        assert isinstance(defined_regions, regionmask.core.regions.Regions)
        self.defined_regions = defined_regions

        mask = defined_regions.mask(lon, lat)
        self.regions_land_sea = mask.values
        self.regions = mask.values[self.idx_l]
        self.region_names = defined_regions.abbrevs

        assert isinstance(compression_factor, float)

        self.compression_factor = compression_factor

        ## initialise dictionary for regional lon lat points, scaling, wavelet and geometry coefficients

        self.C = {}
        self.lamb = {}
        self.d = {}
        self.pairs = {}
        self.pair_types = {}
        self.max_iter = {}

        ## initialise dictionary for reconstruction

        self.compressed_wavelets = {}

    def _get_lonlat(self, reg):

        lons = self.lons[self.idx_l]
        lats = self.lats[self.idx_l]

        lons_reg = lons[self.regions == reg]
        lats_reg = lats[self.regions == reg]

        return lats_reg, lons_reg

    def get_region_with_buffer(self, reg):

        lats_reg, lons_reg = self._get_lonlat(reg)

        x0 = self.grid_resolution
        y0 = self.grid_resolution
        x1 = self.grid_resolution * 2
        y1 = self.grid_resolution * 2

        if lons_reg.max() + self.grid_resolution > self.lons.max():
            x1 = self.grid_resolution
        if lons_reg.min() - self.grid_resolution < self.lons.min():
            x0 = 0
        if lats_reg.max() + self.grid_resolution > self.lats.max():
            y1 = self.grid_resolution
        if lats_reg.min() - self.grid_resolution < self.lats.min():
            y0 = 0

        lons_box, lats_box = np.meshgrid(
            np.arange(lons_reg.min() - x0, lons_reg.max() + x1, 2.5),
            np.arange(lats_reg.min() - y0, lats_reg.max() + y1, 2.5),
        )

        mask_box = self.defined_regions.mask(lons_box, lats_box).values

        ## get indices of sea area
        idx_box = np.logical_and(
            np.isin(self.lons, lons_box), np.isin(self.lats, lats_box)
        )
        land_sea_mask_box = self.idx_l[idx_box].reshape(
            mask_box.shape[0], mask_box.shape[1]
        )

        ## get indices of surrounding region
        buffer_region = land_sea_mask_box & (mask_box != reg)
        buffer_full_grid = idx_box & (self.regions_land_sea != reg)

        sea_mask_box = ~land_sea_mask_box & (mask_box == reg)
        land_mask_box = land_sea_mask_box & (mask_box == reg)

        ## get only buffer region and land grid points
        buffer_and_land = np.logical_or(buffer_region, land_mask_box)

        return (
            lats_box,
            lons_box,
            land_sea_mask_box,
            buffer_full_grid[self.idx_l],
            sea_mask_box,
            land_mask_box,
            buffer_and_land,
        )

    def _check_for_odd_cluster(self, cluster):

        if len(cluster) % 2 == 0:
            return False
        else:
            return True

    def _group_geo_neighbours(self, reg, it):

        lats, lons, land_sea_mask_box, _, _, _, _ = self.get_region_with_buffer(reg)
        lats = lats[land_sea_mask_box]
        lons = lons[land_sea_mask_box]

        if it == 0:
            ## initialise the first indices of x values
            idx_initial_x = np.arange(len(lats))

        else:
            idx_initial_x = np.hstack(([pair[0] for pair in self.pairs[reg][it - 1]]))
            lats = lats[idx_initial_x]

        ## make cut based on lat, and then find closest neighbors for odd left out pieces based on lon

        lats_reg = np.sort(np.unique(lats))
        pairs = []
        pair_types = []

        i_count = 0
        for i_lat in lats_reg:

            idx_lat = np.atleast_1d(
                idx_initial_x[np.squeeze(np.argwhere(lats == i_lat))]
            )

            if not self._check_for_odd_cluster(idx_lat):
                ## column-wise pairs are made, first row = x, second row = y
                pairs.append(idx_lat.reshape(-1, 2).T)
                pair_types.append("even")

            else:

                if len(idx_lat) > 3:
                    pairs.append(idx_lat[:-3].reshape(-1, 2).T)
                    pair_types.append("even")
                    pairs.append(idx_lat[-3:])
                    pair_types.append("odd")

                elif len(idx_lat) == 3:
                    pairs.append(np.array(idx_lat))
                    pair_types.append("odd")

                else:

                    if i_lat != lats_reg[-1]:
                        ## roll over latitude to next point
                        lats[np.squeeze(np.argwhere(lats == i_lat))] = lats_reg[
                            i_count + 1
                        ]

                    else:
                        ## unless we are at last latitude in which case merge to pair before
                        if pair_types[-1] == "odd":
                            pair_types[-1] = "even"
                            pairs.append(
                                np.array([pairs[-1][-1], idx_lat[0]]).reshape(2, -1)
                            )
                            pair_types.append("even")
                            pairs[-2] = pairs[-2][:-1].reshape(2, -1)
                        else:
                            ## either cut off last group into odd pair or if only one group make odd
                            if pairs[-1].shape[1] != 1:
                                pairs.append(
                                    np.array(
                                        [pairs[-1][0, -1], pairs[-1][1, -1], idx_lat[0]]
                                    )
                                )
                                pair_types.append("odd")
                                pairs[-2] = pairs[-2][:, :-1]
                            else:
                                pairs[-1] = np.array(
                                    [pairs[-1][0, -1], pairs[-1][1, -1], idx_lat[0]]
                                )
                                pair_types[-1] = "odd"

            i_count += 1

        self.pairs[reg].append(pairs)
        self.pair_types[reg].append(pair_types)

        return

    def _get_full_group(self, reg, it, x0):

        members = [x0]

        for i_count in np.arange(it + 1)[::-1]:

            for pair, pair_type in zip(
                self.pairs[reg][i_count], self.pair_types[reg][i_count]
            ):

                for i_mem in set(members):

                    if i_mem in pair:

                        if pair_type == "even":

                            members.append(pair[0, np.argwhere(pair == i_mem)[0][1]])
                            members.append(pair[1, np.argwhere(pair == i_mem)[0][1]])

                        else:

                            members.append(pair[0])
                            members.append(pair[1])
                            members.append(pair[2])

        return np.unique(np.squeeze(np.stack(members)))

    def _split(self, reg, it):

        self._group_geo_neighbours(reg, it)

        return

    def _predict(self, reg, it):

        ## need to perform operation on temporal medians, make sure original regional grid is maintained
        ## throughout iterations

        (
            _,
            _,
            _,
            buffer_full_grid_l,
            _,
            _,
            buffer_and_land,
        ) = self.get_region_with_buffer(reg)

        d = np.full([self.X.shape[0], buffer_and_land.sum()], np.nan)
        lam = np.full(buffer_and_land.sum(), np.nan)

        if it == 0:

            mask_reg_buffered = np.logical_or(buffer_full_grid_l, (self.regions == reg))
            vals = self.X[:, mask_reg_buffered]

        else:

            vals = self.C[reg][it - 1]

        for pair, pair_type in zip(self.pairs[reg][it], self.pair_types[reg][it]):

            if pair_type == "even":

                ## fill in regression errors for y, i.e., wavelet coefficients
                d[:, pair[1, :]] = vals[:, pair[1, :]] - vals[:, pair[0, :]]

                k = 0
                for x_val, y_val in zip(pair[0, :], pair[1, :]):

                    ## if full group size is odd than lam = ((group_size-1)/2)/group_size else lam=0.5
                    group_size = len(self._get_full_group(reg, it, x_val))

                    lam[y_val] = (it + 1) / group_size

            else:

                d[:, pair[1]] = vals[:, pair[1]] - vals[:, pair[0]]
                d[:, pair[2]] = vals[:, pair[2]] - vals[:, pair[0]]

                group_size = len(self._get_full_group(reg, it, pair[0]))

                lam[pair[1]] = (it + 2) / group_size
                lam[pair[2]] = (it + 2) / group_size

        self.d[reg].append(d)
        self.lamb[reg].append(lam)

        return

    def _update(self, reg, it):

        ## calculate C and return in matrix form of size len(pairs)
        (
            _,
            _,
            _,
            buffer_full_grid_l,
            _,
            _,
            buffer_and_land,
        ) = self.get_region_with_buffer(reg)

        c = np.full([self.X.shape[0], buffer_and_land.sum()], np.nan)
        if it == 0:
            mask_reg_buffered = np.logical_or(buffer_full_grid_l, (self.regions == reg))
            vals = self.X[:, mask_reg_buffered]

        else:

            vals = self.C[reg][it - 1]

        for pair, pair_type in zip(self.pairs[reg][it], self.pair_types[reg][it]):

            if pair_type == "even":

                for x_val, y_val in zip(pair[0, :], pair[1, :]):

                    members = self._get_full_group(reg, it, x_val)

                    c[:, members] = np.repeat(
                        (
                            vals[:, x_val]
                            + self.lamb[reg][it][y_val] * self.d[reg][it][:, y_val]
                        ).reshape(-1, 1),
                        len(members),
                        axis=1,
                    )

            else:

                assert self.lamb[reg][it][pair[1]] == self.lamb[reg][it][pair[2]]

                members = self._get_full_group(reg, it, pair[0])

                c[:, members] = np.repeat(
                    (
                        vals[:, pair[0]]
                        + self.lamb[reg][it][pair[1]]
                        * (self.d[reg][it][:, pair[1]] + self.d[reg][it][:, pair[2]])
                    ).reshape(-1, 1),
                    len(members),
                    axis=1,
                )

        self.C[reg].append(c)
        return

    def execute(self, reg):

        ##initialise coefficient lists for region
        self.pairs[reg] = []
        self.pair_types[reg] = []
        self.C[reg] = []
        self.lamb[reg] = []
        self.d[reg] = []

        it = 0

        M_cut = (self.regions == reg).sum() // 2

        self.max_iter[reg] = np.inf

        while it < self.max_iter[reg]:

            self._split(reg, it)
            self._predict(reg, it)
            self._update(reg, it)

            if len(self.pairs[reg][it]) == 1:
                self.max_iter[reg] = it
                # print('Terminated after %i iterations for region, %s'%(self.max_iter[reg], self.region_names[reg]))

            it += 1

        return

    def execute_over_regions(self):

        # Parallel(n_jobs=10)(delayed(self.execute)(int(reg)) for reg in tqdm(np.unique(self.regions)))
        offset = 0
        counter = 0
        for reg in np.unique(self.regions[~np.isnan(self.regions)]):
            self.execute(int(reg))

            counter += 1

        return

    def summary_plot(self, reg, it, lat_pc, lon_pc, t=-1):

        (
            lats,
            lons,
            _,
            buffer_full_grid_l,
            _,
            _,
            buffer_and_land,
        ) = self.get_region_with_buffer(reg)
        mask_reg_buffered = np.logical_or(buffer_full_grid_l, (self.regions == reg))
        x0 = lons.min() - self.grid_resolution
        x1 = lons.max() + self.grid_resolution
        y0 = lats.min() - self.grid_resolution
        y1 = lats.max() + self.grid_resolution

        if x0 < self.lons.min():
            x0 = self.lons.min()
        if y0 < self.lats.min():
            y0 = self.lats.min()
        if x1 > self.lons.max():
            x1 = self.lons.max()
        if y1 > self.lats.max():
            y1 = self.lats.max()

        titles = ["Groups", "Wavelet Coefficients", "Scaling Coefficients"]
        colors = ["tab20c", "cool", "YlGnBu"]
        fig = plt.figure(figsize=(12, 8))
        axs = []

        axs.append(
            fig.add_subplot(1, 3, 1, projection=ccrs.Robinson(central_longitude=0))
        )
        axs.append(
            fig.add_subplot(1, 3, 2, projection=ccrs.Robinson(central_longitude=0))
        )
        axs.append(
            fig.add_subplot(1, 3, 3, projection=ccrs.Robinson(central_longitude=0))
        )

        for i_ax, ax in enumerate(axs):

            y_ma = np.zeros(self.idx_l.shape)
            y_ma = np.ma.masked_array(y_ma, mask=self.idx_l == False)

            if i_ax == 0:

                j = np.full(self.idx_l.sum(), np.nan)
                k = np.zeros((buffer_and_land).sum())

                count = 1
                for pair, pair_type in zip(
                    self.pairs[reg][it], self.pair_types[reg][it]
                ):

                    if pair_type == "even":

                        for x_val in pair[0, :]:

                            members = self._get_full_group(reg, it, x_val)
                            k[members] = count
                            count += 1

                    else:

                        members = self._get_full_group(reg, it, pair[0])
                        k[members] = count

                        count += 1

                j[mask_reg_buffered] = k

            elif i_ax == 1:

                j = np.full(self.idx_l.sum(), np.nan)
                j[mask_reg_buffered] = self.d[reg][it][t, :]

            elif i_ax == 2:

                j = np.full(self.idx_l.sum(), np.nan)
                j[mask_reg_buffered] = self.C[reg][it][t, :]

            y_ma[self.idx_l] = j
            ax.coastlines()
            mesh = ax.pcolormesh(
                lon_pc,
                lat_pc,
                y_ma,
                cmap=colors[i_ax],
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            ax.set_extent([x0, x1, y0, y1])
            ax.set_title(titles[i_ax], fontsize=12)
            plt.colorbar(mesh, ax=ax, shrink=0.8, aspect=35, orientation="horizontal")

    def _compress(self, d, scale=True):

        if scale:

            X_cluster = MinMaxScaler().fit_transform(np.abs(d.reshape(-1, 1)))

            X_cluster = X_cluster.reshape(d.shape[0], d.shape[1])

        else:

            X_cluster = d

        idx_dom = (X_cluster >= self.compression_factor).any(axis=0)

        d_comp = np.zeros_like(d)
        d_comp[:, idx_dom] = d[:, idx_dom]

        return d_comp

    def compress_wavelet_coeff(self, reg, scale=False, compress=False):

        """
        Retrieve sparse representation of spatial patterns
        """

        d_final = np.zeros_like(self.d[reg][0])

        for it in range(self.max_iter[reg] + 1):

            nan_mask = ~np.isnan(self.d[reg][it])
            d_final[nan_mask] = self.d[reg][it][nan_mask]

        if compress:

            self.d[reg].append(self._compress(d_final, scale=scale))

        else:

            self.d[reg].append(d_final)

        return

    def find_nearest(self, centres, value):

        idx = np.argsort(np.abs(centres - value))

        return idx[:10].reshape(1, -1)

    def find_global_nearest(self, label):

        values, counts = np.unique(label, return_counts=True)

        ## get most common values that then defines our neighbourhood
        values = values[np.argsort(counts)[::-1]][:10]

        return values.reshape(1, -1)

    def sample_from_nearest(self, reg, d_all, n_samp, d_prop=None):

        seed = 4861946401452

        if d_prop == None:

            # multivariate_shrinkage = LedoitWolf().fit(d_all_sel)
            mean = np.mean(d_all, axis=0)  # multivariate_shrinkage.location_
            cov = np.cov(d_all, rowvar=False)  # multivariate_shrinkage.covariance_
        else:
            mean = d_prop[0][1:]
            cov = d_prop[1]

        d_sampled = np.random.Generator(np.random.PCG64(seed)).multivariate_normal(
            mean, cov, n_samp, method="eigh"
        )
        if d_prop != None:
            d_sampled = np.c_[np.zeros([n_samp, 1]), d_sampled]

        return d_sampled

    def sample_wavelet_coeff(
        self,
        y,
        n_samp,
        parallel=False,
        output_dsampled=False,
        d_prop=None,
        labels_ind=None,
        compute_neighbourhood=True,
        **kwargs
    ):

        """
        Sample prototype of spatial patterns from clustered groups using Y_t

        Process
        -------

        Get labels at each time step, then find and sample from timesteps corresponding to the label

        kwargs are for compress_wavelet_coeff function

        """
        all_regs = (
            self.C.keys()
        )  # np.unique(self.regions[~np.isnan(self.regions)]).astype(int)
        offset = 0
        if list(all_regs)[0] != 0:
            offset = -1 * list(all_regs)[0]

        ## first check whether wavelets are compressed and stored
        # labels = {}
        d_samples = {}
        d_all = {}

        if compute_neighbourhood:
            labels = np.hstack(
                (
                    [
                        np.vstack(
                            (
                                [
                                    self.find_nearest(self.C[reg][-1][:, 0], yval)
                                    for yval in y[reg + offset]
                                ]
                            )
                        )
                        for reg in all_regs
                    ]
                )
            )

            labels = np.vstack(([self.find_global_nearest(label) for label in labels]))

        for reg in all_regs:

            _, _, _, _, _, _, buffer_and_land = self.get_region_with_buffer(reg)
            d_samples[reg] = np.zeros([n_samp, y[0].shape[0], buffer_and_land.sum()])

            if len(self.d[reg]) != self.max_iter[reg] + 2:

                self.compress_wavelet_coeff(reg, **kwargs)

            ## Note: if compute neighbourhood is True then d_prop is automatically overriden as None
            if compute_neighbourhood:

                # labels = np.vstack(([self.find_nearest(self.C[reg][-1][:,0], yval) for yval in y[reg+offset]]))
                d_all[reg] = np.stack(([self.d[reg][-1][label, :] for label in labels]))
                d_prop_reg = np.full([len(labels)], None)

            else:

                labels = labels_ind

                d_all[reg] = np.empty([len(labels)])
                d_prop_reg = d_prop[reg]

            # labels = np.swapaxes(np.stack((labels)),0,1) ##want rows to be each timestep
            if parallel:

                d_samp_temp = Parallel(n_jobs=10)(
                    delayed(self.sample_from_nearest)(
                        reg,
                        d_all[reg][counter],
                        n_samp,
                        d_prop=d_prop_reg[counter],
                        **kwargs
                    )
                    for counter in range(len(labels))
                )

                for counter, _ in enumerate(labels):

                    d_samples[reg][:, counter, :] = d_samp_temp[counter]

            else:

                for counter, _ in enumerate(labels):

                    d_samples[reg][:, counter, :] = self.sample_from_nearest(
                        reg,
                        d_all[reg][counter],
                        n_samp,
                        d_prop=d_prop_reg[counter],
                        **kwargs
                    )
        if output_dsampled:

            return [d_samples, d_all, labels]

        else:

            return d_samples

    def inverse_predict_and_update(self, reg, it, d_sampled, y):

        _, _, _, _, _, _, buffer_and_land = self.get_region_with_buffer(reg)
        cinv = np.full([y.shape[0], buffer_and_land.sum()], np.nan)

        for pair, pair_type in zip(self.pairs[reg][it], self.pair_types[reg][it]):

            if pair_type == "even":

                for x_val, y_val in zip(pair[0, :], pair[1, :]):

                    if it == 0:

                        cinv[:, x_val] = (
                            y[:, x_val]
                            - self.lamb[reg][it][y_val] * d_sampled[:, y_val]
                        )

                        cinv[:, y_val] = cinv[:, x_val] + d_sampled[:, y_val]

                    else:

                        members_x = self._get_full_group(reg, it - 1, x_val)
                        members_y = self._get_full_group(reg, it - 1, y_val)

                        cinv[:, members_x] = np.repeat(
                            (
                                y[:, x_val]
                                - self.lamb[reg][it][y_val] * d_sampled[:, y_val]
                            ).reshape(-1, 1),
                            len(members_x),
                            axis=1,
                        )

                        cinv[:, members_y] = np.repeat(
                            (cinv[:, x_val] + d_sampled[:, y_val]).reshape(-1, 1),
                            len(members_y),
                            axis=1,
                        )
            else:

                assert self.lamb[reg][it][pair[1]] == self.lamb[reg][it][pair[2]]

                if it == 0:

                    cinv[:, pair[0]] = y[:, pair[0]] - self.lamb[reg][it][pair[1]] * (
                        d_sampled[:, pair[1]] + d_sampled[:, pair[2]]
                    )

                    cinv[:, pair[1]] = cinv[:, pair[0]] + d_sampled[:, pair[1]]

                    cinv[:, pair[2]] = cinv[:, pair[0]] + d_sampled[:, pair[2]]

                else:

                    members_x = self._get_full_group(reg, it - 1, pair[0])
                    members_1 = self._get_full_group(reg, it - 1, pair[1])
                    members_2 = self._get_full_group(reg, it - 1, pair[2])

                    cinv[:, members_x] = np.repeat(
                        (
                            y[:, pair[0]]
                            - self.lamb[reg][it][pair[1]]
                            * (d_sampled[:, pair[1]] + d_sampled[:, pair[2]])
                        ).reshape(-1, 1),
                        len(members_x),
                        axis=1,
                    )

                    cinv[:, members_1] = np.repeat(
                        (cinv[:, pair[0]] + d_sampled[:, pair[1]]).reshape(-1, 1),
                        len(members_1),
                        axis=1,
                    )

                    cinv[:, members_2] = np.repeat(
                        (cinv[:, pair[0]] + d_sampled[:, pair[2]]).reshape(-1, 1),
                        len(members_2),
                        axis=1,
                    )

        return cinv

    def sample_and_update(self, reg, Y, Cinv, d_temp, sample):

        for it in np.arange(self.max_iter[reg] + 1)[::-1]:

            if it not in Cinv.keys():

                Cinv[it] = []

            if it == self.max_iter[reg]:

                Cinv[it].append(
                    self.inverse_predict_and_update(reg, it, d_temp, Y.reshape(-1, 1))
                )

            else:

                y_temp = Cinv[it + 1][sample]

                Cinv[it].append(
                    self.inverse_predict_and_update(reg, it, d_temp, y_temp)
                )

        return

    def ground(self, reg, Y, D_sampled, n_samp=100, **kwargs):

        (
            lats_box,
            lons_box,
            land_sea_mask_box,
            buffer_full_grid_l,
            sea_mask_box,
            land_mask_box,
            buffer_and_land,
        ) = self.get_region_with_buffer(reg)

        Cinv = {}

        D_temp = D_sampled[reg]

        for sample in np.arange(n_samp):

            self.sample_and_update(reg, Y, Cinv, D_temp[sample], sample)

        Cinv_final = np.stack((Cinv[0]))

        Cinv_temp = np.zeros(
            [
                Cinv_final.shape[0],
                Cinv_final.shape[1],
                buffer_and_land.shape[0],
                buffer_and_land.shape[1],
            ]
        )
        Cinv_temp[:, :, buffer_and_land] = Cinv_final

        return Cinv_temp[:, :, buffer_and_land]

    def ground_over_regions(
        self,
        Y_all,
        n_samp=100,
        progress=False,
        parallel=True,
        d_prop=None,
        labels_ind=None,
        output_dsampled=False,
        compute_neighbourhood=True,
        **kwargs
    ):

        ## Assume all regional time series are of equal length
        all_regs = self.C.keys()
        grounded_grid = np.full(
            [n_samp, Y_all[0].shape[0], len(all_regs), self.idx_l.sum()], np.nan
        )

        start_time = time.time()
        D_sampled = self.sample_wavelet_coeff(
            Y_all,
            n_samp,
            parallel=False,
            output_dsampled=output_dsampled,
            d_prop=d_prop,
            labels_ind=labels_ind,
            compute_neighbourhood=compute_neighbourhood,
        )

        if output_dsampled:

            labels = D_sampled[2]
            D_handover = D_sampled[1]
            D_sampled = D_sampled[0]

        print(
            "Time taken for sampling wavelets ----%s secs----"
            % (time.time() - start_time)
        )

        offset = 0
        if list(all_regs)[0] != 0:
            offset = -1 * list(all_regs)[0]

        start_time = time.time()
        if parallel:
            if progress:
                Cinvs = Parallel(n_jobs=10)(
                    delayed(self.ground)(
                        int(reg),
                        Y_all[int(reg) + offset],
                        D_sampled,
                        n_samp=n_samp,
                        **kwargs
                    )
                    for reg in tqdm(all_regs)
                )

            else:

                Cinvs = Parallel(n_jobs=10)(
                    delayed(self.ground)(
                        int(reg),
                        Y_all[int(reg) + offset],
                        D_sampled,
                        n_samp=n_samp,
                        **kwargs
                    )
                    for reg in all_regs
                )
                print(
                    "Time taken for grounding ----%s secs----"
                    % (time.time() - start_time)
                )

            for reg in all_regs:

                (
                    lats,
                    lons,
                    _,
                    buffer_full_grid_l,
                    _,
                    _,
                    buffer_and_land,
                ) = self.get_region_with_buffer(reg)
                mask_reg_buffered = np.logical_or(
                    buffer_full_grid_l, (self.regions == reg)
                )

                grounded_grid[:, :, int(reg) + offset, mask_reg_buffered] = Cinvs[
                    int(reg) + offset
                ]

        else:

            Cinvs = []

            for reg in all_regs:
                (
                    lats,
                    lons,
                    _,
                    buffer_full_grid_l,
                    _,
                    _,
                    buffer_and_land,
                ) = self.get_region_with_buffer(reg)
                mask_reg_buffered = np.logical_or(
                    buffer_full_grid_l, (self.regions == reg)
                )

                Cinvs.append(
                    self.ground(
                        int(reg),
                        Y_all[int(reg) + offset],
                        D_sampled,
                        n_samp=n_samp,
                        **kwargs
                    )
                )

                grounded_grid[:, :, int(reg) + offset, mask_reg_buffered] = Cinvs[
                    int(reg) + offset
                ]

            print(
                "Time taken for grounding ----%s secs----" % (time.time() - start_time)
            )

        if output_dsampled:
            return np.nanmean(grounded_grid, axis=2), D_handover, labels

        else:
            return np.nanmean(grounded_grid, axis=2)
