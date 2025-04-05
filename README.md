
Lifting scheme
===
![DOI](https://img.shields.io/badge/DOI-arXiv%3A2501.04018-hotpink)
![PRs](https://img.shields.io/badge/PRs-welcome-green
)
![PRs](https://img.shields.io/badge/regionmask-violet
)

## Workflow

The lifting scheme is ana adaption of wavelet-based image compression to irregularly-shaped (i.e., non-square or -rectangular) grids. For a given input grid it iteratively performs 3 steps till the grid is compressed to a single value called the *scaling coefficient*. The three steps are:
1. **Split**: split the grid into pairs of 2 (and in case of an odd number of grid cells one triplet will exist).
2. **Predict**: Assign x-y values to each pair/triple and then store the differences of x-y as the *wavelet coefficients*
3. **Update**: update each pair/triple group's values with that of their mean.
![Slide1](https://github.com/snath-xoc/Lifting/blob/master/images/Slide1.png)

## Example usage

Our adaption of this for spatio-temporal frameworks means that we can compress irregular images into a single value across many different time-steps/samples. In modelling complex phenomena such as weather and climate this allows one to extract a regional signal from which to model large-scale responses.

For example one can compress the image of India as shown below for a single time step

![UKESM1-0-LL_SAS_0_July_GIF_lift](https://github.com/snath-xoc/Lifting/blob/master/images/lifting_SAS_slow.gif)

And again for the Mediterranean

![UKESM1-0-LL_MED_0_July_GIF_lift](https://github.com/snath-xoc/Lifting/blob/master/images/lifting_MED_slow.gif)

## Use cases

Amongst others, the lifting scheme has been used for climate model emulation. It's compression of multiple climate fields allows efficient joint, multivariate emulation (see [MERCURY](https://arxiv.org/abs/2501.04018)). 






