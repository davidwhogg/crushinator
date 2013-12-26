import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def flux_cumtrapz(np.ndarray[DTYPE_t, ndim=2] sed,
    np.ndarray[DTYPE_t, ndim=1] filter, int N):

    cdef int i
    cdef double dh, dl
    cdef double fh, fl
    cdef double v

    v = 0
    for i in range(N - 1):
        dh = sed[i + 1, 0]
        dl = sed[i, 0]
        fh = sed[i + 1, 1] * filter[i + 1]
        fl = sed[i, 1] * filter[i]
        v += (dh - dl) * (fh + fl)
    return 0.5 * v