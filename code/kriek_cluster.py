import os
import numpy as np

from scipy.interpolate import interp1d

def load_filters():
    """
    Return the synthetic filter profile and array of the filter
    wavelengths.
    """
    fname = '../data/newfirm/cosmos_seds/kriek/approx_synthetic_filter.dat'
    d = np.loadtxt(fname)

    # magic
    start = 3.06818586175
    space = 0.07358558

    # make wavelength dict
    filter_loglams = np.zeros((22, d[:, 0].size))
    for i in range(22):
        filter_loglams[i] = d[:, 0] + start + i * space

    return filter_loglams, d[:, 1]

def make_measurements(sed_loglam, sed_flam, filter_loglams, filter_profile):
    """
    Make flux measurements for sed.
    """
    # cubic is too effing slow, should be dense enough to be ok
    f = interp1d(sed_loglam, sed_flam, kind='linear', bounds_error=False,
                 fill_value=0.0)

    fluxes = np.zeros(filter_loglams.shape[0])
    for i in range(filter_loglams.shape[0]):
        flam = f(filter_loglams[i])
        fluxes[i] = np.sum(flam * filter_profile)

    return fluxes

def get_distance_metrics(fluxes):
    """
    Return array of distance metrics defined in 
    Kriek et al. (2011).   Not restricted to where 
    there is overlaping coverage...
    """
    b_values = np.zeros((fluxes.shape[0], fluxes.shape[0]))
    for i in range(fluxes.shape[0]):
        a = np.sum(fluxes[i] * fluxes, axis=1)
        a /= np.sum(fluxes ** 2., axis=1)
        b_values[i] = np.sum((fluxes[i] - a[:, None] * fluxes) ** 2., axis=1)
        b_values[i] /= np.sum(fluxes[i] ** 2.)

    return b_values

def get_clusters(b_values, tol, min_num=20):
    """
    Assign clusters according to Kriek et al. (2011)
    """
    clusters = np.zeros(b_values.shape[0]) - 1.
    parents = np.array([], dtype=np.int)

    while True:
        N_analogs = np.zeros_like(clusters)
        for i in range(N_analogs.size):
            ind = np.where((b_values[i] < tol) & (b_values[i] > 0.0) & 
                           (clusters == -1.))[0]
            N_analogs[i] = ind.size

        if np.all(N_analogs < min_num):
            break

        new_parent = np.where(N_analogs == np.max(N_analogs))[0][0]
        parents = np.append(parents, new_parent)
        ind = np.where((b_values[new_parent] < tol) &
                       (b_values[new_parent] > 0.0))

        clusters[ind] = new_parent

    return clusters

def refine_clusters(b_values, clusters):
    """
    Given rough cluster assignments, check whether better
    assignments exist.
    """
    Nchanged = 0
    parents = np.unique(clusters.astype(np.int))[1:]
    for i in range(b_values.shape[0]):
        c = clusters[i].astype(np.int)
        if c == -1:
            continue

        vals = b_values[i, parents]
        current = b_values[i, c]

        ind = np.where((vals < current) & (vals > 0.0))[0]

        if ind.size > 0:
            mn = vals[ind].min()
            Nchanged += 1
            ind = vals == mn
            clusters[i] = parents[ind]

    print 'Reassigned %d galaxies' % Nchanged
    return clusters
        


if __name__ == '__main__':

    base = '../data/newfirm/cosmos_seds/kriek/'
    
    if False:
        # load filters
        filter_loglams, filter_profile = load_filters()

        # get sed paths
        os.system('ls ' + base + 'sed* > foo.txt')
        f = open('foo.txt')
        sed_files = f.readlines()
        f.close()
        os.system('rm foo.txt')

        N = len(sed_files)
        fluxes = np.zeros((N, filter_loglams.shape[0]))
        for i in range(N):
            sed = np.loadtxt(sed_files[i][:-1])
            fluxes[i] = make_measurements(np.log10(sed[:, 0]), sed[:, 1],
                                          filter_loglams, filter_profile)

            np.savetxt(base + 'synth_fluxes.dat', fluxes)
    else:
        fluxes = np.loadtxt(base + 'synth_fluxes.dat')
    print 'b_values'
    b_values = get_distance_metrics(fluxes)

    print 'clusters'
    # assign initial clusters
    # 0.003 give 32 unique
    while True:
        tol = np.float(raw_input('tolerace --> '))
        clusters = get_clusters(b_values, tol)
        print np.unique(clusters), len(np.unique(clusters))
        cmd = raw_input('action --> ')
        if cmd == 'stop':
            break

    f = open(base + 'clusters_%0.3f.txt' % tol, 'w')
    for i in range(clusters.size):
        f.write('%d\n' % clusters[i])
    f.close()

    # refine assignments
    clusters = refine_clusters(b_values, clusters)

    f = open(base + 'refined_clusters_%0.3f.txt' % tol, 'w')
    for i in range(clusters.size):
        f.write('%d\n' % clusters[i])
    f.close()
