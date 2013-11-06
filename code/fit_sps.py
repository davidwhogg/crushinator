import os
import sys
import numpy as np

from fit_sps_utils import *

base = '../data/newfirm/cosmos_seds/kriek/'

# central wavelengths, indicies, and filter names
cws, pht_inds, err_inds, col_names, filter_names = get_info()

# normalization factors
factors = 25. + 5 * np.log10(cws) + 2.406
factors = 10 ** (-0.4 * factors + 18)

# redshift catalog
zcat = np.loadtxt('../data/newfirm/cosmos-1.deblend.redshifts/' +
                  'cosmos-1.deblend.v5.1.zout')

# photometric catalog
pcat = np.loadtxt('../data/newfirm/cosmos-1.deblend.v5.1.cat')

# get list of seds
os.system('ls ' + base + 'sed_* > foo')
f = open('foo')
sed_list = f.readlines()
os.system('rm foo')

N = 10
for i in range(N):
    # get sed and identity
    filename = sed_list[i][:-1]
    sed = np.loadtxt(filename)
    l, u = string.rfind(filename, '_'), string.rfind(filename, '.')
    ident = np.int(filename[l + 1: u])

    # get redshift
    ind = np.where(zcat[:, 0] == ident)[0]
    if zcat[ind, 1] != -1:
        z = zcat[ind, 1]
    else:
        z = zcat[ind, 5]
    sed[:, 0] *= (1. + z)

    # compute sed flux densities
    filters, idx = regrid_filters(filter_names, sed[:, 0])
    sed_fluxes = compute_flux_densities(sed, filters[idx])

    # data
    ind = np.where(pcat[:, 0] == ident)[0]
    fluxes = pcat[ind, pht_inds] * factors
    ferrs = pcat[ind, err_inds] * factors

    # only use filters that fall on model sed
    print filters.shape, len(idx), cws.shape
    fluxes = fluxes[idx]
    ferrs = ferrs[idx]
    waves = cws[idx]

    # only use filters with measurements
    ind = fluxes > 0
    sed_fluxes = sed_fluxes[ind]
    fluxes = fluxes[ind]
    ferrs = ferrs[ind]
    waves = cws[ind]

    print filename
    f = pl.figure()
    pl.plot(np.log10(sed[:, 0]), sed[:, 1])
    pl.plot(np.log10(waves), sed_fluxes, 'o')
    f.savefig('../plots/foo.png')
    if i==1:
        assert 0

    # fit
    scale, chi2 = fit_flux_densities(fluxes, ferrs, sed_fluxes)
    print i, chi2, chi2 / waves.size

    # plot if desired
    if sys.argv[1] == 'True':
        f=pl.figure()
        pl.plot(np.log10(sed[:, 0]), sed[:, 1] * scale)
        pl.plot(np.log10(waves), sed_fluxes * scale, 'o')
        pl.errorbar(np.log10(waves), fluxes, yerr=ferrs, fmt='ro')
        pl.title('$\chi^2 = %0.2f$' % (chi2 / waves.size))
        f.savefig('../plots/sed_fits_%d.png' % ident)
