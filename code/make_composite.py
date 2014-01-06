import numpy as np

def get_scales(synth_fluxes):
    """
    Find the scales to best match the 
    synthetic photometry, randomly selecting the primary.
    """
    N = synth_fluxes.shape[0]
    prime_ind = np.random.randint(N)

    primary = np.atleast_2d(synth_fluxes[prime_ind]).T

    scales = np.ones(N)
    for i in range(N):
        if i == prime_ind:
            continue
        
        a = np.atleast_2d(synth_fluxes[i]).T
        rh = np.dot(a.T, primary)
        lh = np.dot(a.T, a)
        scales[i] = rh / lh

    return scales

def shift_scale(z, cws, scales, fluxes):
    """
    Shift and scale the observed data.
    """
    waves = cws[None, :] / (1. + z[:, None])
    scaled = fluxes * scales[:, None]
    return waves, scaled


if __name__ == '__main__':

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

    os.system('ls ' + base + 'sed* > foo.txt')
    f = open('foo.txt')
    files = f.readlines()
    f.close()
    os.system('rm foo.txt')

    # get redshift and fluxes
    z = np.zeros(len(files))
    fluxes = np.zeros((len(files), len(pht_inds)))
    for i in range(len(files)):
        ident = np.float(files[i].split('_')[-1][:-5])
        ind = np.where(ident == pcat[:, 0])[0]

        if zcat[ind, 1] != -1:
            z[i] = zcat[ind, 1]
        else:
            z[i] = zcat[ind, 5]

        fluxes[i] = pcat[ind, pht_inds] * factors

    # synthetic fluxes
    synth_fluxes = np.loadtxt(base + 'synth_fluxes.dat')

    # scales from fits to data
    fit_results = np.loadtxt(base + 'fit_results.dat')
    synth_fluxes *= fit_results[:, 0][:, None]

    # load clusters
    clusters = np.loadtxt(base + 'refined_clusters_0.003.txt')
    u = np.unique(clusters)
    for i in range(u.size):
        if u[i] == -1.:
            continue
        N = np.where(clusters==u[i])[0].size

        # get scales
        ind = (clusters == u[i])
        scales = get_scales(synth_fluxes[ind])
 
        # shift and scale
        shifted_waves, scaled_fluxes = shift_scale(z[ind], cws, 
                                                   scales, fluxes[ind])

        # plot
        a = 0.2
        s = 2. * np.sqrt(200. / N)
        factor = 2.5

        f = pl.figure()
        for j in range(shifted_waves.shape[0]):
            pl.plot(np.log10(shifted_waves[j]), scaled_fluxes[j], 
                    'ko', alpha=a, ms=s)
        pl.xlim(np.log10(1200), np.log10(50000.))
        pl.title('$N = %d$, $z = %0.2f\pm%0.2f$' % (N, np.mean(z[ind]), 
                                              np.std(z[ind])))
        ind = (scaled_fluxes > 0)
        pl.ylim(0, np.mean(scaled_fluxes[ind]) *  factor)
        pl.ylabel('Scaled $F_\lambda\, (erg\,cm^{-2}\,s^{-1}\,\AA^{-1})$')
        pl.xlabel('$\log_{10}(\lambda / \AA)$')
        f.savefig('../plots/kriek/shifted_and_scaled_refined_%d.png' % i)
