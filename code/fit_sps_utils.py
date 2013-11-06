import os
import string
import numpy as np
import matplotlib.pyplot as pl

from scipy.interpolate import interp1d

def load_filter_names(filename):
    f = open(filename)
    l = f.readlines()
    f.close()

    for i in range(len(l)):
        l[i] = l[i].split('/')[1][:-1]

    return np.array(l)

def get_info():

    base = '../data/newfirm/cosmos_seds/kriek/'

    # names of filters
    filter_names = load_filter_names(base + 'filter_names.txt')

    # central wavelengths, and indicies in photometric catalog
    d = np.loadtxt('../data/newfirm/cosmos_central_and_indicies.dat')
    cws, pht_inds, err_inds = d[:, 0], d[:, 1].astype(np.int), \
        d[:, 2].astype(np.int)

    # not using Kaper or 24um
    cws = cws[1:-2]
    pht_inds = pht_inds[1:-2]
    err_inds = err_inds[1:-2]

    # catalog columns
    f = open('../data/newfirm/cosmos-1.deblend.v5.1.cat')
    col_names = np.array(f.readlines()[0][1:].split())
    f.close()

    return cws, pht_inds, err_inds, col_names, filter_names

def compute_flux_densities(sed, filters):
    fls = np.zeros(filters.shape[0])
    for i, f in enumerate(filters):
        fls[i] = np.sum(sed[:, 1] * filters[i]) / np.sum(filters[i])
        
    return fls

def approx_flux_densities(cws, sed):
    fls = np.zeros(cws.shape[0])
    for i in range(cws.shape[0]):
        ind = np.argsort(np.abs(sed[:, 0] - cws[i]))
        fls[i] = np.median(sed[ind[:4], 1])
    return fls

def regrid_filters(filter_names, sps_grid):

    f = open('../data/newfirm/FILTER.RES.v7.R300')
    l = np.array(f.readlines())
    f.close()

    idx = []
    new_profiles = np.zeros((filter_names.size, sps_grid.size))
    for i in range(len(filter_names)):
        for j, line in enumerate(l):
            line = line.split()
            if line[1].split('/')[-1] == filter_names[i]:
                N = np.int(line[0])
                old_lambda = np.zeros(N)
                old_profile = np.zeros(N)
                for k in range(N):
                    line = l[j + 1 + k].split()
                    old_lambda[k] = np.float(line[1])
                    old_profile[k] = np.float(line[2])
                f = interp1d(old_lambda, old_profile, kind='cubic',
                             bounds_error=False, fill_value=0.0)
 
                new_profiles[i] = f(sps_grid)
                ind = new_profiles[i] < 0.0
                new_profiles[i][ind] = 0.0
                if (np.any(new_profiles[i]!=0)) & (new_profiles[i, 0]==0) & \
                        (new_profiles[i,-1]==0):
                    idx.append(i)
                break

    return new_profiles, idx

def fit_flux_densities(data, var, model):
    data = np.atleast_2d(data).T
    model = np.atleast_2d(model).T
    invar = np.diag(1. / var)
    
    rh = np.dot(model.T, np.dot(invar, data))
    lh = np.dot(model.T, np.dot(invar, model))

    scale = np.dot(np.linalg.inv(lh), rh)[0]
    chi2 = np.sum((data - model * scale) ** 2. * invar)

    return scale, chi2

if __name__ == '__main__':

    base = '../data/newfirm/cosmos_seds/kriek/'

    cws, pht_inds, err_inds, col_names, filter_names = get_info()

    filename = base + 'sed_10024.dat'
    sed = np.loadtxt(filename)

    l, u = string.rfind(filename, '_'), string.rfind(filename, '.')
    ident = np.int(filename[l + 1: u])
    
    zcat = np.loadtxt('../data/newfirm/cosmos-1.deblend.redshifts/cosmos-1.deblend.v5.1.zout')
    ind = np.where(zcat[:, 0] == ident)[0]
    if zcat[ind, 1] != -1:
        z = zcat[ind, 1]
    else:
        z = zcat[ind, 5]

    sed[:, 0] *= (1. + z)
    
    filters, idx = regrid_filters(filter_names, sed[:, 0])

    
    fls = compute_flux_densities(sed, filters[idx])
    apx = approx_flux_densities(cws[idx], sed)

    cat = np.loadtxt('../data/newfirm/cosmos-1.deblend.v5.1.cat')
    ind = np.where(cat[:, 0] == ident)[0]
    fluxes = cat[ind, pht_inds]
    ferrs = cat[ind, err_inds]

    factors = 25. + 5 * np.log10(cws) + 2.406
    factors = 10 ** (-0.4 * factors + 18)
    fluxes *= factors
    ferrs *= factors
    fluxes = fluxes[idx]
    ferrs = ferrs[idx]
    cws = cws[idx]

    s, c = fit_flux_densities(fluxes, ferrs ** 2., fls)
    print s, c

    ind = (fluxes > 0) & (cws < 50000.)

    f=pl.figure()
    pl.plot(np.log10(sed[:, 0]), sed[:, 1] * s)
    pl.plot(np.log10(cws), fls * s, 'bo')
    pl.plot(np.log10(cws), apx * s, 'go')
    pl.errorbar(np.log10(cws[ind]), fluxes[ind], yerr=ferrs[ind], fmt='ro')
    f.savefig('../plots/foo.png')
