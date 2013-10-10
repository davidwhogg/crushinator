import numpy as np
import matplotlib.pyplot as pl

from matplotlib.ticker import FuncFormatter, FixedLocator
from scipy.special import erf

def xtick(x, p):
    return '%0.1f' % (10. ** x)

def error_func(x, factor):
    return 0.5 * (erf(ftr * 0.5 * np.sqrt(np.pi) * x) + 1)

# magic numbers that align profile to data
# lifted from Kriek et al. (2011), Fig. 1
ftr = 250
width = 0.074
offset = 0.009

# profile segments
N = 5000
x1 = np.linspace(-0.02, 0.035, N)  
profile1 = error_func(x1 - offset, ftr)
x3 = x1 + width
profile3 = 1. - profile1
x2 = np.linspace(x1.max(), x3.min(), N / 2.)
profile2 = np.ones_like(x2)

# profile
x = np.append(x1, np.append(x2, x3))
profile = np.append(profile1, np.append(profile2, profile3))

# start so first filter = 10 at 1250 \AA
start = 3.06818586175
# spacing roughly corresponding to Kriek et al.
space = 0.07358558

# save
out = np.zeros((x.size, 2))
out[:, 0] = x
out[:, 1] = profile
np.savetxt('../data/newfirm/cosmos_seds/kriek/approx_synthetic_filter.dat',
           out)

# ticks
fmt = FuncFormatter(xtick)
locs = np.log10(np.array([1250., 2500., 5000., 10000.,
                          20000., 40000.]))
lct = FixedLocator(locs)

# plot
f = pl.figure(figsize=(20, 2))
ax = pl.axes()
for i in range(22):
    ax.plot(x + start + i * space, profile, 'r')
ax.xaxis.set_major_formatter(fmt)
ax.xaxis.set_major_locator(lct)
pl.xlim(np.log10(1150.), np.log10(51500.))
pl.ylim(0.0, 1.1)
f.savefig('../plots/foo.png')

