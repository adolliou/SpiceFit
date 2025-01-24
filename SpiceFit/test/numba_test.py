from numba import jit
import numpy as np

@jit(nopython=True, inline='always', cache=True)
def fitting_function(x, I0, x0, s0, a0):
    m = 0
    sum = np.zeros(len(x), dtype=np.float64)

    sum = sum + I0 * np.exp(- ((x0 - x) ** 2) / (2 * (s0) ** 2))

    sum = sum + a0 * x ** (0)

    return sum

if __name__ == '__main__':
    x = np.linspace(0, 10, 1000, dtype=np.float64)

    y = fitting_function(x, I0=5, x0=5, s0=1, a0=0.2)