import os as os
import numpy as np
from scipy.optimize import curve_fit

def gauss(x, a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2.*sigma**2))


def red_sequence_fit(mag,color,maglim):

    mask = (mag<22.)

    color2 = color[mask]
    n,c    = np.histogram(color2,30)      
    c      = (c+(c[1]-c[0])*0.5)[:-1]
    
    err   = np.ones(len(c))

    fit_gauss = curve_fit(gauss,c,n,sigma=err,absolute_sigma=True)
    a         = fit_gauss[0][0]
    mu        = fit_gauss[0][1]
    sigma     = abs(fit_gauss[0][2])
    
    mask   = (color < (mu+3.*sigma))*(color > (mu-3.*sigma))

    return mask,mu,sigma
