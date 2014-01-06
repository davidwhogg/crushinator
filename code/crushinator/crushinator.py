import numpy as np

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
                 eps=1e6):
        assert redshifts is not None, 'z inference not supported yet'

        self.count = 0

        self.N = fluxes.shape[0]
        self.Nfilters = fluxes.shape[1]

        self.eps = eps
        self.sed = initial_sed
        self.amps = np.ones(self.N)
        self.fluxes = fluxes
        self.models = np.zeros((self.N, self.Nfilters))
        self.fix_amps = False
        self.wave_grid = self.sed[:, 0]
        self.flux_errors = flux_errors
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

        # add more checks...
        self.run_checks()

    def run_checks(self):
        """
        Check input.
        """
        m = 'Flux shape does not match'
        assert self.fluxes.shape == self.flux_errors.shape, m

    def filter_calcs(self, filterfiles):
        """
        Calculate filter quantities, and interpolation funcs.
        """
        self.filters, self.eff_waves = load_filters(filterfiles)
        self.interp_funcs = build_filter_interp(self.filters)
        self.max_waves = [self.filters[i][:, 0].max() 
                          for i in range(len(self.filters.keys()))]

    def fit_datum(self, model, fluxes, flux_errors):
        """
        Fit one objects set of fluxes.
        """
        lh = np.sum(model ** 2. / flux_errors ** 2.)
        rh = np.sum(model * fluxes / flux_errors ** 2.)
        scale = rh / lh
        chi2 = np.sum((fluxes - scale * model) ** 2. / flux_errors ** 2.)
        return scale, chi2

    def optimize(self, max_iter=1000):
        """
        Infer the MAP sed.
        """
        # change to something better...
        if self.fix_redshifts:
            p0 = self.initial_sed[:, 1].copy()
            args = ()

        result = fmin_bfgs(self.loss, p0, args=args, maxiter=max_iter)
        self.sed = result

    def loss(self, p):
        """
        Return the value of the loss for the current model.
        """
        p /= p.sum()
        sed = np.vstack((self.wave_grid, p)).T
        self.models = np.zeros_like(self.fluxes)
        for i in range(self.N):
            self.models[i] = compute_fluxes(self.interp_funcs, sed,
                                            self.redshifts[i], self.max_waves,
                                            self.filters)
            if not self.fix_amps:
                self.amps[i], c = self.fit_datum(self.models[i],
                                                 self.fluxes[i],
                                                 self.flux_errors[i])

        nll = 0.5 * (self.fluxes - self.amps[:, None] * self.models) ** 2. /\
            self.flux_errors ** 2.
        nll = nll.sum()
        reg = self.eps * np.sum((p[1:] - p[:-1]) ** 2.)
        if self.count % 1000 == 0:
            print nll, reg, nll + reg

        self.count += 1
        return nll + reg
