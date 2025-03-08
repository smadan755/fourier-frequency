def clean_complex(x, tol=1e-10):
    result = x.copy()
    result.real[abs(result.real) < tol] = 0
    result.imag[abs(result.imag) < tol] = 0


    return result