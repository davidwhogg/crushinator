import numpy as np
import multiprocessing

from scipy.optimize import fmin_bfgs

from .utils import compute_fluxes, effective_wavelength
from .utils import load_filters, build_filter_interp

class Crushinator(object):
    """
    Infer the underlying spectral energy distribution 
    from photometry.
    """
    def __init__(self, fluxes, flux_errors, filterfiles, initial_sed, 
                 redshifts=None, known_amps=None, ini_amps=None, zmax=4.,
                 outlier=False, eps=1e6):
        assert redshifts is not None, 'z inference not supported yet'

        self.count = 0

        self.N = fluxes.shape[0]
        self.Nfilters = fluxes.shape[1]

        self.eps = eps
        self.sed = initial_sed
        self.amps = np.ones(self.N)
        self.fluxes = fluxes
        self.models = np.zeros((self.N, self.Nfilters))
        self.outlier = outlier
        self.wave_grid = self.sed[:, 0]
        self.flux_vars = flux_errors ** 2.
        self.initial_sed = initial_sed

        self.filter_calcs(filterfiles)

        if redshifts is None:
            self.redshifts = np.random.rand(self.N) * zmax
        else:
            self.redshifts = redshifts
            self.fix_redshifts = True

        if ini_amps is not None:
            self.amps = ini_amps

        if known_amps is not None:
            self.amps = known_amps
            self.fix_amps = True
        else:
            self.fix_amps = False

        # add more checks...
        self.run_checks()

    def run_checks(self):
        """
        Check input.
        """
        m = 'Flux shape does not match'
        assert self.fluxes.shape == self.flux_vars.shape, m

    def filter_calcs(self, filterfiles):
        """
        Calculate filter quantities, and interpolation funcs.
        """
        self.filters, self.eff_waves = load_filters(filterfiles)
        self.interp_funcs = build_filter_interp(self.filters)
        self.max_waves = [self.filters[i][:, 0].max() 
                          for i in range(len(self.filters.keys()))]

    def fit_datum(self, model, fluxes, flux_vars):
        """
        Fit one objects set of fluxes.
        """
        lh = np.sum(model ** 2. / flux_vars)
        rh = np.sum(model * fluxes / flux_vars)
        scale = rh / lh
        chi2 = np.sum((fluxes - scale * model) ** 2. / flux_vars)
        return scale, chi2

    def optimize(self, max_iter=1000, outlier_init_logprior=-6,
                 outlier_init_logscale=0):
        """
        Infer the MAP sed.
        """
        # change to something better...
        if self.fix_redshifts:
            p0 = self.initial_sed[:, 1].copy()
            args = ()

        # outlier initialization
        if self.outlier:
            p0 = np.append(p0, [outlier_init_logprior, outlier_init_logscale])

        result = fmin_bfgs(self.loss, p0, args=args, maxiter=max_iter)
        self.sed = result

    def loss(self, p):
        """
        Return the value of the loss for the current model.
        """
        if self.outlier:
            f = np.exp(p[-2])
            bad_vars = self.flux_vars * (1. + np.exp(p[-1]))
            bad_prior = 1. / (1.+ f)
            good_prior = 1. - bad_prior
            p = p[:-2]

        p /= p.sum()
        sed = np.vstack((self.wave_grid, p)).T

        # models
        self.models = np.zeros_like(self.fluxes)
        for i in range(self.N):
            self.models[i] = compute_fluxes(sed, self.redshifts[i],
                                            self.max_waves, self.filters)

        # Amplitudes
        if not self.fix_amps:
            for i in range(self.N):
                self.amps[i], c = self.fit_datum(self.models[i],
                                                 self.fluxes[i],
                                                 self.flux_vars[i])

        sqe = (self.fluxes - self.amps[:, None] * self.models) ** 2.

        if self.outlier:
            # ghetto log-sum-exp
            ag = 1. / np.sqrt(2. * np.pi * self.flux_vars)
            ab = 1. / np.sqrt(2. * np.pi * bad_vars)
            gl = ag * np.exp(-0.5 * sqe / self.flux_vars)
            bl = ab * np.exp(-0.5 * sqe / bad_vars)
            gl = np.sum(gl)
            bl = np.sum(bl)

            nll = -np.log(good_prior * gl + bad_prior * bl)
        else:
            nll = np.sum(0.5 * sqe / self.flux_vars)

        reg = self.eps * np.sum((p[1:] - p[:-1]) ** 2.)
        if self.count % 20 == 0:
            print nll, reg, nll + reg

        self.count += 1
        return nll + reg
