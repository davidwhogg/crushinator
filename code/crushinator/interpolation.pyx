import numpy as np
cimport numpy as np
from cinterp cimport *
cimport cython

from libc.stdlib cimport malloc, free

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def interp(np.ndarray[DTYPE_t, ndim=1] xref, np.ndarray[DTYPE_t, ndim=1] yref,
           np.ndarray[DTYPE_t, ndim=1] xout, np.ndarray[DTYPE_t, ndim=1] yout,
           int Nref, int Nout, int kind):

    cdef int i
    cdef double *x = <double *>malloc(Nref * sizeof(double))
    cdef double *y = <double *>malloc(Nref * sizeof(double))

    for i in range(Nref):
        x[i] = xref[i]
        y[i] = yref[i]

    cdef gsl_interp_accel *acc
    acc = gsl_interp_accel_alloc()

    cdef gsl_spline *spline

    if kind == 0:
        spline = gsl_spline_alloc(gsl_interp_linear, Nref)

    if kind == 1:
        spline = gsl_spline_alloc(gsl_interp_cspline, Nref)

    gsl_spline_init(spline, x, y, Nref)

    for i in range(Nout):
        if ((xout[i] >= xref[0]) & (xout[i] <= xref[Nref - 1])):
            yout[i] = gsl_spline_eval(spline, xout[i], acc)


    gsl_spline_free(spline)
    gsl_interp_accel_free(acc)

    free(x)
    free(y)
