
import numpy as np
from numba import jit



def fitting_function(x, I0,x0,s0,I1,x1,s1,a0):
	m = 0
	sum = np.zeros(len(x), dtype=np.float64)
	
	
	sum = sum + I0 * np.exp( - ( ( x0 - x )**2 ) / (2 * ( s0 )**2 ))

	sum = sum + I1 * np.exp( - ( ( x1 - x )**2 ) / (2 * ( s1 )**2 ))

	sum = sum + a0 * x ** (0)

	return sum




def jacobian_function(x, I0,x0,s0,I1,x1,s1,a0):
	m = np.zeros((len(x), 7), dtype=np.float64)
	sum = np.zeros(len(x), dtype=np.float64)
	
	
	m[:, 0] = 1 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[:, 1] = ((x - x0)/(s0**2)) * I0 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[:, 2] = s0 * (((x - x0)**2)/(s0**3)) * I0 * np.exp(- ((x - x0)**2)/(2 * s0**2)) 
	m[:, 3] = 1 * np.exp(- ((x - x1)**2)/(2 * s1**2)) 
	m[:, 4] = ((x - x1)/(s1**2)) * I1 * np.exp(- ((x - x1)**2)/(2 * s1**2)) 
	m[:, 5] = s1 * (((x - x1)**2)/(s1**3)) * I1 * np.exp(- ((x - x1)**2)/(2 * s1**2)) 
	m[:, 6] = x**0

	return m