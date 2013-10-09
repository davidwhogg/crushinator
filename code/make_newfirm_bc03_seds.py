import os
import subprocess
import numpy as np

# newfirm sps data file
filename = '../data/newfirm/cosmos-1.deblend.sps/' + \
    'cosmos-1.bc03.del.deblend.v5.1.fout'
parms = np.loadtxt(filename)

# newfirm photometry data file
phot_cat = np.loadtxt('../data/newfirm/cosmos-1.deblend.v5.1.cat')

# newfirm redshift data file
filename = '../data/newfirm/cosmos-1.deblend.redshifts/' + \
    'cosmos-1.deblend.v5.1.zout'
z_cat = np.loadtxt(filename)

# run through data to check id and assign redshift
z = np.zeros(z_cat.shape[0])
for i in range(z.size):
    assert ((parms[i, 0] == z_cat[i, 0]) &
            (phot_cat[i, 0] == z_cat[i, 0]))

    # check for spec redshift
    if z_cat[i, 1] != -1:
        z[i] = z_cat[i, 1]
    else:
        z[i] = z_cat[i, 5]

# Select galaxies, using criteria similar to Kriek et al. (2011)
# This gives 3076 galaxies, a bit less than Kriek et al.
ind = np.where((parms[:, 1] != -1) &    # sps exists
               (phot_cat[:, -1] == 1) & # the `use` flag
               (z > 0.5) &              # redshift criterion
               (z < 2.0) &              # redshift criterion
               (phot_cat[:, 22] / phot_cat[:, 23] > 25.))[0] # S/N

parms = parms[ind]
print 'There are %d galaxies with required criteria' % \
    ind.size

# bc03 library directory
library_dir = '../external/bc03/models/Padova1994/chabrier/'

# A_v = 1.086 * tau_v
N = parms.shape[0]
for i in range(N):
    ident, z, tau = parms[i, :3]
    metal, age, Av = parms[i, 3:6]

    # only using the 'm62' bc03
    assert metal == 0.02

    # input for csp_galaxev
    commands = library_dir + 'bc2003_hr_m62_chab_ssp.ised\n'
    commands += 'Y\n%0.5f\n'  % (Av / 1.086)
    commands += '1.0\n4\n%0.5f\n20.\n' % 10. ** (tau - 9.)
    commands += 'out\n'

    # run galaxev
    p = subprocess.Popen('../external/bc03/src/csp_galaxev', 
                         stdin=subprocess.PIPE)
    p.communicate(input=commands)

    # input for galaxev
    commands = 'out.ised\n1200, 50000, 7000, 1, %0.2f\n' % z
    commands += '%0.5f\n' % 10. ** (age - 9.)
    commands += '../data/newfirm/cosmos_seds/kriek/sed_%d.dat\n' % int(ident)

    # run galaxev to get sed
    p = subprocess.Popen('../external/bc03/src/galaxevpl', 
                         stdin=subprocess.PIPE)
    p.communicate(input=commands)

    # cleanup
    os .system('rm out*')
