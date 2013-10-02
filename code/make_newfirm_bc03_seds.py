import os
import numpy as np

filename = '../data/newfirm/cosmos-1.deblend.sps/' + \
    'cosmos-1.bc03.del.deblend.v5.1.fout'

parms = np.loadtxt(filename)
ind = np.where(parms[:, 1] != -1)[0]
parms = parms[ind]
print 'There are %d galaxies with measured parameters' % \
    ind.size

# silly bc03 cant find library outside directory
os.system('cp ../external/bc03/models/Padova1994/chabrier/*62*ised .')

# A_v = 1.086 * tau_v
N = 30
for i in range(N):
    if parms[i, 1] != -1:
        ident, z, tau = parms[i, :3]
        metal, age, Av = parms[i, 3:6]
        
# cleanup
os.system('rm *ised')
