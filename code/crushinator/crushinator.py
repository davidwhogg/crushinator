import numpy as np

from scipy.optimize import fmin_bfgs
from .utils import compute_fluxes, effective_wavelength

class Crushinator(object):
    """
    Infer the underlying spectral energy distribution 
    from photometry.
    """
    def __init__(self, fluxes, flux_errors, filters, initial_sed, 
                 redshifts=None, known_amps=None, zmax=4., eps=1e6):
        assert known_amps is not None, 'amp inference not supported yet'
        assert redshifts is not None, 'z inference not supported yet'

        self.N = fluxes.shape[0]
        self.Nfilters = fluxes.shape[1]

        self.eps = eps
        self.sed = initial_sed
        self.fluxes = fluxes
        self.models = np.zeros((self.N, self.Nfilters))
        self.filters = filters
        self.wave_grid = self.sed[:, 0]
        self.flux_errors = flux_errors
        self.initial_sed = initial_sed

        self.eff_waves = np.zeros(self.Nfilters)
        for i in range(self.Nfilters):
            self.eff_waves[i] = effective_wavelength(filters[i])

        if redshifts is None:
            self.redshifts = np.random.rand(self.N) * zmax
        else:
            self.redshifts = redshifts
            self.fix_redshifts = True

        if known_amps is not None:
            self.amps = known_amps
            self.fix_amps = True

        self.run_checks()
        self.optimize()

    def run_checks(self):
        """
        Check input.
        """
        m = 'Flux shape does not match'
        assert self.fluxes.shape == self.flux_errors.shape, m

        m = 'Filters should be a dictionary of 2D numpy arrays'
        assert type(self.filters) is dict, m

    def fit_datum(self, model, fluxes, flux_errors):
        """
        Fit one objects set of fluxes.
        """
        lh = np.sum(model ** 2. / flux_errors ** 2.)
        rh = np.sum(model * fluxes / flux_errors ** 2.)
        scale = rh / lh
        chi2 = np.sum((fluxes - scale * model) ** 2. / flux_errors ** 2.)
        return scale, chi2

    def optimize(self):
        """
        Infer the MAP sed.
        """
        # change to something better...
        if self.fix_redshifts:
            p0 = self.initial_sed[:, 1]
            args = ()

        result = fmin_bfgs(self.loss, p0, args=args, maxiter=10)
        self.sed = result

    def loss(self, p):
        """
        Return the value of the loss for the current model.
        """
        sed = np.vstack((self.wave_grid, p)).T
        self.models = np.zeros_like(self.fluxes)
        for i in range(self.N):
            self.models[i] = compute_fluxes(self.filters, sed,
                                            self.redshifts[i])

        nll = 0.5 * (self.fluxes - self.amps[:, None] * self.models) ** 2. /\
            self.flux_errors ** 2.
        nll = nll.sum()
        reg = self.eps * np.sum(np.gradient(p) ** 2.)
        print nll, reg, nll + reg
        return nll + reg
