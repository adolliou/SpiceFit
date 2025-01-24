
import numpy as np
from numba import jit



def fitting_function(x, I0,x0,s0,a0):
	m = 0
	sum = np.zeros(len(x), dtype=np.float64)
	
	
	sum = sum + I0 * np.exp( - ( ( x0 - x )**2 ) / (2 * ( s0 )**2 ))

	sum = sum + a0 * x ** (0)

	return sum




def jacobian_function(x, I0,x0,s0,a0):
	m = np.zeros((1, 4), dtype=np.float64)
	sum = np.zeros(len(x), dtype=np.float64)
	
	
	m[0, 0] = 1 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[0, 1] = ((x - x0)/(s0**2)) * I0 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[0, 2] = s0 * (((x - x0)**2)/(s0**3)) * I0 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[0, 3] = x**0

	return m