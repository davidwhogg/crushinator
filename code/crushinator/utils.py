import numpy as np

from .interpolation import interpolation
from flux_calculation import flux_cumtrapz 

from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

def load_filters(filterfiles):
    """
    Return dictionary of filters.
    """
    filters = {}
    Nfilters = len(filterfiles)
    eff = np.zeros(Nfilters)
    for i in range(Nfilters):
        filters[i] = np.loadtxt(filterfiles[i])
        norm = simps(filters[i][:, 1], filters[i][:, 0])
        filters[i][:, 1] /= norm
        eff[i] = effective_wavelength(filters[i])

    return filters, eff

def effective_wavelength(filt):
    """
    Calculate pivot wavelength
    """
    a = np.sum(filt[:, 0] * filt[:, 1])
    b = np.sum(filt[:, 1] / filt[:, 0])
    return np.sqrt(a / b)

def compute_one_flux(filt, sed):
    """
    Compute the flux in a single filter, given the redshifted sed.
    """
    curve = np.zeros_like(sed[:, 0])
    interpolation(filt[:, 0], filt[:, 1], sed[:, 0],
                  curve, filt.shape[0], curve.shape[0], 0)

    return flux_cumtrapz(sed, curve, curve.size)

def compute_fluxes(sed, redshift, max_waves, filters):
    """
    Compute fluxes in bandpasses, given redshift and sed
    """
    tmp_sed = sed.copy()
    tmp_sed[:, 0] = (1. + redshift) * tmp_sed[:, 0]

    fluxes = np.zeros(len(filters.keys()))
    for i in range(fluxes.size):
        if tmp_sed[0, 0] < max_waves[i]:
            fluxes[i] = compute_one_flux(filters[i], tmp_sed)
        else:
            fluxes[i] = 0.0

    return fluxes

def build_filter_interp(filters):
    """
    Make the interpolation functions for filters.
    """
    interp_funcs = {}
    for i in range(len(filters.keys())):
        norm = simps(filters[i][:, 1], filters[i][:, 0])
        assert np.allclose(norm, 1.0), \
            'Filter %d is not normalized' % j

        interp_funcs[i] = interp1d(filters[i][:, 0], filters[i][:, 1],
                                   bounds_error=False, fill_value=0.0,
                                   kind='cubic')
    return interp_funcs
        
def shift_and_scale_model(fluxes, zs, eff_lambdas, max_wave, Ngrid=None,
                          flux_errs=None, zerrs=None, amps=None,
                          wave_grid=None, scl=0.5, bandwidth=None,
                          est_window=10):
    """
    Return a model sed by shifting and scaling the data.
    
    TO DO: account for missing amplitudes, redshift errors
    """
    shifted = eff_lambdas[None, :] / (1. + zs)[:, None]
    if wave_grid is None:
        if Ngrid is None:
            Ngrid = np.round(shifted.shape[0] * scl)
        wave_grid = np.linspace(np.log(shifted.min()), np.log(shifted.max()),
                                Ngrid)
        dlt = wave_grid[1] - wave_grid[0]
        bins = wave_grid - dlt / 2.
        bins = np.append(bins, wave_grid[-1] + dlt / 2.)
        wave_grid = np.exp(wave_grid)
        bins = np.exp(bins)
    else:
        dlt = np.log(wave_grid[1]) - np.log(wave_grid[0])

    if amps is None:
        # to do
        pass
    else:
        scaled = fluxes * amps[:, None]
        sed = np.zeros_like(wave_grid)
        for i in range(sed.size):
            ind = np.where((shifted >= bins[i]) & 
                           (shifted < bins[i + 1]))
            if ind[0].size > 0:
                if flux_errs is None:
                    sed[i] = np.median(scaled[ind])
                else:
                    sed[i] = np.sum(scaled[ind] / flux_errs[ind])
                    sed[i] /= np.sum(1. / flux_errs[ind])
            else:
                sed[i] = np.nan

    # fill in nans
    ind = np.where(sed == sed)
    interp = interp1d(wave_grid[ind], sed[ind])
    ind = np.where(sed != sed)
    sed[ind] = interp(wave_grid[ind])

    # extend sed to end of reddest filter by fitting a power-law
    assert max_wave > wave_grid[-1]
    ext = np.array([np.log(wave_grid[-1]) + dlt])
    while np.exp(ext[-1]) < max_wave:
        ext = np.append(ext, ext[-1] + dlt)
    ext = np.exp(ext)
    d = np.log(sed[-est_window:])
    w = wave_grid[-est_window:]
    a = np.vstack((w, np.ones_like(w)))
    rh = np.dot(a, d)
    lh = np.dot(a, a.T)
    v = np.dot(np.linalg.inv(lh), rh)
    sed = np.append(sed, np.exp(v[0] * ext + v[1]))
    wave_grid = np.append(wave_grid, ext)

    # smooth if desired
    if bandwidth is not None:
        sed = gaussian_filter1d(sed, sigma=bandwidth)

    sed = np.vstack((wave_grid, sed)).T
    return shifted, scaled, sed
