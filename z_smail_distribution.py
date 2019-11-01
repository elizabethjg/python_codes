import numpy as np

def N(data,p2,p3):
	z,p1 = data
	return (z**p1)*np.exp(-1*(z/p2)**p3)
	
def N_factor(data,factor,p2,p3):
	z,p1 = data
	return factor*(z**p1)*np.exp(-1*(z/p2)**p3)
