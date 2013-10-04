import os
import subprocess
import numpy as np

# bc03 library directory
library_dir = '../external/bc03/models/Padova1994/chabrier/'

# newfirm data file
filename = '../data/newfirm/cosmos-1.deblend.sps/' + \
    'cosmos-1.bc03.del.deblend.v5.1.fout'

# read data
parms = np.loadtxt(filename)
ind = np.where(parms[:, 1] != -1)[0]
parms = parms[ind]
print 'There are %d galaxies with measured parameters' % \
    ind.size

# A_v = 1.086 * tau_v
N = parms.shape[0]*0 + 1
for i in range(N):
    ident, z, tau = parms[i, :3]
    metal, age, Av = parms[i, 3:6]

    # only using the 'm62' bc03
    assert metal == 0.02

    # input for csp_galaxev
    commands = library_dir + 'bc2003_hr_m62_chab_ssp.ised\n'
    commands += 'Y\n%0.3f\n'  % (Av / 1.086)
    commands += '0.3\n4\n%0.2f\n20.\n' % 10. ** (age - 9.)
    commands += 'out\n\n'

    # run galaxev
    p = subprocess.Popen('../external/bc03/src/csp_galaxev', 
                         stdin=subprocess.PIPE)
    p.communicate(input=commands)

    # input for galaxev
    commands = 'out.ised\n1200, 50000, 7000, 1, 0\n'
    commands += '0.3\n' % 10. ** (age - 9.)
    commands += '../data/newfirm/cosmos_seds/sed_%d.dat\n' % int(ident)

    # run galaxev to get sed
    p = subprocess.Popen('../external/bc03/src/galaxevpl', 
                         stdin=subprocess.PIPE)
    p.communicate(input=commands)

    # cleanup
    os .system('rm out*')
