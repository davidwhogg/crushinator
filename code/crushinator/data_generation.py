import numpy as np

from .utils import *
from scipy.integrate import simps
from scipy.interpolate import interp1d

def make_data(N, sedfile, filterfiles, noise_level=0.01, zmin=0.0, zmax=2.,
              Nzgrid=1000, outlier_frac=0.05, outlier_scale=10.):
    """
    Generate fake data given the sed and filters.  `filterfiles` should
    be a list of strings.
    """
    sed = np.loadtxt(sedfile)
    filters, eff = load_filters(filterfiles)
    interp_funcs = build_filter_interp(filters)
    max_waves = [filters[i][:, 0].max() for i in range(len(filters.keys()))]

    # from faintest bin in Ilbert et al. (2009)
    va = 0.126
    vb = 4.146
    vc = 5.925
    zg = np.linspace(0.0, zmax, Nzgrid)
    fz = (zg ** va + zg ** (va * vb))
    fz /= (zg ** vb + vc)
    fz /= simps(fz, zg)
    z_interp = interp1d(zg, fz)
    redshifts = np.zeros(N) - 99.
    for i in range(N):
        while redshifts[i] < 0.:
            r = np.random.rand()
            z = np.random.rand() * (zmax - zmin) + zmin
            p = z_interp(z)
            if r < p:
                redshifts[i] = z

    fluxes = np.zeros((N, len(filterfiles)))
    for i in range(N):
        fluxes[i] = compute_fluxes(sed, redshifts[i], max_waves, filters)

    # add option for non-trivial amplitudes here.
    amplitudes = np.ones(N)
    fluxes *= amplitudes[:, None]

    Noutlier = np.int(fluxes.shape[0]) * outlier_frac

    # noise
    noise_level = np.ones_like(fluxes) * np.median(fluxes) * noise_level
    if Noutlier > 0:
        nl = noise_level.copy()
        nl[:Noutlier] = noise_level[:Noutlier] * outlier_scale
    else:
        nl = noise_level
    noise = np.random.randn(fluxes.shape[0], fluxes.shape[1]) * nl
    fluxes += noise

    return redshifts, fluxes, amplitudes, noise_level, eff, filters

def make_cosmos_like_data(N, sedfile, filterfiles, noise_level=1.0, zmax=2.,
                          zmin=0.05, Ndraw=10000, Nzgrid=1000, minmag=21,
                          maxmag=25):
    """
    Generate fake data with similar i-band mags and errors given the sed and 
    filters.  `filterfiles` should be a list of strings.  Errors are assigned 
    using the order [u, g, r, i, z] mod the filter number.
    """
    Nfilters = len(filterfiles)
    sed = np.loadtxt(sedfile)
    filters, eff = load_filters(filterfiles)

    # apparent magnitudes
    imag_dist = np.loadtxt('../data/cosmos_imag_hist.dat')
    f = interp1d(imag_dist[:, 0], imag_dist[:, 1])
    imags = np.array([])
    while imags.size < N:
        tm = np.random.rand(Ndraw) * (maxmag - minmag) + minmag
        rs = np.random.rand(Ndraw)
        vs = rs - f(tm)
        ind = np.where(vs < 0)
        imags = np.append(imags, tm[ind])
        if imags.size > N:
            imags = imags[:N]

    # apparent magnitude error
    mag_err_dists = np.loadtxt('../data/cosmos_err_vs_imag.dat')
    mag_errs = np.zeros((N, Nfilters))
    for i in range(Nfilters):
        interp = interp1d(mag_err_dists[:, 0], mag_err_dists[:, i + 1])
        mag_errs[:, i] = interp(imags)

    # redshifts, magic from Ilbert et al. (2009)
    redshifts = np.zeros(N) - 99.
    ms = np.linspace(22.25, 24.75, 6)
    a = np.array([0.497, 0.488, 0.372, 0.273, 0.201, 0.126])
    b = np.array([12.64, 9.251, 6.736, 5.281, 4.494, 4.146])
    c = np.array([0.381, 0.742, 1.392, 2.614, 3.932, 5.925])
    zg = np.linspace(0.0, zmax, Nzgrid)
    a_interp = interp1d(ms, a)
    b_interp = interp1d(ms, b)
    c_interp = interp1d(ms, c)
    for i in range(N):
        if imags[i] < ms[0]:
            va, vb, vc = a[0], b[0], c[0]
        elif imags[i] > ms[-1]:
            va, vb, vc = a[-1], b[-1], c[-1]
        else:
            va = a_interp(imags[i])
            vb = b_interp(imags[i])
            vc = c_interp(imags[i])
        fz = (zg ** va + zg ** (va * vb))
        fz /= (zg ** vb + vc)
        fz /= simps(fz, zg)
        z_interp = interp1d(zg, fz)
        while redshifts[i] < 0.:
            r = np.random.rand()
            z = np.random.rand() * (zmax - zmin) + zmin
            p = z_interp(z)
            if r < p:
                redshifts[i] = z

    # distance modulii
    dm_dist = np.loadtxt('../data/WMAP7_dm.dat')
    interp = interp1d(dm_dist[:, 0], dm_dist[:, 1])
    dms = interp(redshifts)

    # amplitudes
    amplitudes = 10. ** (-0.4 * (imags + dms))

    # fluxes from sed
    fluxes = np.zeros((N, Nfilters))
    for i in range(N):
        fluxes[i] = compute_fluxes(filters, sed, redshifts[i])

    # match to i band
    fluxes *= (amplitudes / fluxes[:, 3])[:, None]

    # noise
    noise_dist = np.loadtxt('../data/cosmos_err_vs_imag.dat')
    imag_errs = np.zeros_like(fluxes)
    for i in range(Nfilters):
        interp = interp1d(noise_dist[:, 0], noise_dist[:, i + 1])
        imag_errs[:, i] = interp(imags)
        
    flux_errs = 0.4 * np.log(10.) * fluxes * imag_errs * noise_level

    # scale to reasonable numbers
    factor = np.min(fluxes)
    fluxes /= factor
    amplitudes /= factor
    flux_errs /= factor

    # add the noise
    noise = np.random.randn(fluxes.shape[0], fluxes.shape[1]) * flux_errs
    fluxes += flux_errs

    return redshifts, fluxes, amplitudes, flux_errs, eff, filters
