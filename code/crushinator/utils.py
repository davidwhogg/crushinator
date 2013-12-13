import numpy as np

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

def compute_fluxes(filters, sed, redshift):
    """
    Compute fluxes in bandpasses, given redshift and sed
    """
    sed_wave = (1. + redshift) * sed[:, 0]
    sed_flux = sed[:, 1]

    fluxes = np.zeros(len(filters.keys()))
    for i in range(fluxes.size):
        interp = interp1d(filters[i][:, 0], filters[i][:, 1],
                          bounds_error=False, fill_value=0.0)
        if sed_wave[0] < filters[i][:, 0].max():
            curve = interp(sed_wave)
            fluxes[i] = simps(curve * sed_flux, sed_wave)
        else:
            fluxes[i] = 0.0

        if i == 0:
            for j in range(len(filters.keys())):
                norm = simps(filters[j][:, 1], filters[j][:, 0])
                assert np.allclose(norm, 1.0), \
                    'Filter %d is not normalized' % j

    return fluxes
        
def shift_and_scale_model(fluxes, zs, eff_lambdas, Ngrid=None, flux_errs=None,
                          zerrs=None, amps=None, wave_grid=None, scl=0.5,
                          bandwidth=None):
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

    # smooth if desired
    if bandwidth is not None:
        sed = gaussian_filter1d(sed, sigma=bandwidth)

    sed = np.vstack((wave_grid, sed)).T
    return shifted, scaled, sed
