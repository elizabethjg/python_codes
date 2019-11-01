#!/usr/local/anaconda/bin/python
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits

def main(fits_map, threshold, lowlim, highlim, ratio, regular):

	# Read fits file	
	#------------------------------------------------------------------------#
	hdu = fits.open(fits_map)
	hdr = hdu[0].header
	data = hdu[0].data
	hdu.close()
	#------------------------------------------------------------------------#

	# Read image width and height and check if square
	#------------------------------------------------------------------------#
	width = len(data[:,0])#*1.1
	height = len(data[0,:])#*1.1
	if width != height:
		print "ERROR: {} must be a squared image \n".format(fits_map)
		return 
	#------------------------------------------------------------------------#
	
	# Read astrometry from fits header
	#------------------------------------------------------------------------#
	ra0 = hdr['CRVAL1']		# RA of reference pixel in deg 
	dec0 = hdr['CRVAL2']		# Dec of reference pixel in deg
	crpix0 = hdr['CRPIX1']		# X position of ref pixel in fits convention
	crpix1 = hdr['CRPIX2']		# Y position of ref pixel in fits convention
	#------------------------------------------------------------------------#



	# Create grid
	#------------------------------------------------------------------------#
	if regular == False:
		# Define some parameters
		#------------------------------------------------------------------------#
		nlens = int(1. + 6.*sum(np.arange(2.**lowlim) +1.))		# Number of nodes in the hexagonal grid, if we cut each triangle lowlim times
		grid = Grid(nlens)	# Class containing x,y,rc,rcut
		grid.threshold = threshold
		grid.highlim = highlim
		grid.lowlim = lowlim
		grid.ratio = ratio
		#------------------------------------------------------------------------#

		
		grid = multiscale_grid(grid, data)
		print grid.nlens
		# Rescale x, y, rc an rcut
		#------------------------------------------------------------------------#
		grid.x = grid.x[0:grid.nlens]
		grid.y = grid.y[0:grid.nlens]
		grid.rc = grid.rc[0:grid.nlens]
		grid.rcut = grid.rcut[0:grid.nlens]
		grid, sdens0 = build_frr(grid, hdr)
		#------------------------------------------------------------------------#
	else:
		# Define some parameters
		#------------------------------------------------------------------------#
		# nlens = int((width/lowlim)**2)		# Number of nodes in the hexagonal grid, if we cut each triangle lowlim times
		nlens = int(1. + 6.*sum(np.arange(2.**lowlim) +1.))		# Number of nodes in the hexagonal grid, if we cut each triangle lowlim times
		grid = Grid(nlens)	# Class containing x,y,rc,rcut
		grid.threshold = threshold
		grid.highlim = highlim
		grid.lowlim = lowlim
		grid.ratio = ratio
		#------------------------------------------------------------------------#

		print grid.nlens
		grid = regular_grid(grid, width)
		grid.x = grid.x + crpix0
		grid.y = grid.y + crpix1
		grid, sdens0 = build_frr(grid, hdr)
	#------------------------------------------------------------------------#

	print grid.rc

	# """
	# Plot the points and rc radii
	#------------------------------------------------------------------------#
	# circle(grid)
	# plt.plot([x[1:7], x[1]], [y[1:7], y[1]], 'r-', lw=5)
	# plt.savefig('grid_t{}.pdf'.format(grid.threshold))
	#------------------------------------------------------------------------#
	# """


	a = np.zeros(grid.nlens)
	
	f = open('sdens_t{}.dat'.format(grid.threshold), 'w')
	f.write('#REFERENCE 3 {} {} \n'.format(ra0, dec0))
	for i in range(len(grid.x)):
		f.write('{} {} {} {} {} {} \n'.format(i, grid.x[i], grid.y[i], grid.rc[i], grid.rcut[i], a[i]))
	f.close()

	return 0


def regular_grid(grid, width):
	idd = 0
	nmax = int(np.sqrt((grid.nlens))/2)
	rc =  width/pow(2., grid.lowlim+1)
	grid.rc[:] = rc
	grid.rcut[:] = rc*grid.ratio
	for j in np.arange(-nmax, nmax):
		for i in np.arange(-nmax, nmax-np.mod(np.abs(j), 2)):
			grid.x[idd] = i*rc + np.mod(abs(j), 2)*rc/2.
			grid.y[idd] = j*rc*np.sqrt(3.)/2.
			idd+=1
	return grid



def regular_grid2(grid, width):
	idd = 0
	nmax = int(np.sqrt((grid.nlens))/2)
	rc =  grid.lowlim#width/pow(2., grid.lowlim+1)
	grid.rc[:] = rc
	grid.rcut[:] = rc*grid.ratio
	for j in np.arange(-nmax, nmax):
		for i in np.arange(-nmax, nmax-np.mod(np.abs(j), 2)):
			grid.x[idd] = i*rc + np.mod(abs(j), 2)*rc/2.
			grid.y[idd] = j*rc*np.sqrt(3.)/2.
			idd+=1
	return grid


def build_frr(grid, hdr):
	"""
	Convert grid to relative arcsec and build the frr and sdens0 matrix/vector
	"""

	# Scaling kappa <-> absolute mass
	scale = 1
	
	# Convert x, y, rc and rcut from pixel to relative arcsec
	#------------------------------------------------------------------------#
	xpos = np.copy(grid.x)
	ypos = np.copy(grid.y)
	w = wcs.WCS(hdr)
	xabs, yabs = w.all_pix2world(xpos, ypos, 1)
	grid.y = (yabs - hdr['CRVAL2'])*3600.
	grid.x = -(xabs - hdr['CRVAL1'])*3600.*np.cos(hdr['CRVAL2']/180.*np.pi)
	pix2sec = get_pix2sec(hdr)
	grid.rc*=pix2sec
	grid.rcut*=pix2sec
	#------------------------------------------------------------------------#
	
	# Build a matrix of distances r2(i, j) ?		
	#------------------------------------------------------------------------#
	rr = build_distmat(grid)
	frr = sdenspiemd(rr, grid, 1.)
	#------------------------------------------------------------------------#
	sdens0 = np.zeros(grid.nlens)

	"""
	# Read the mass density at the points of the mapping
	#------------------------------------------------------------------------#
	sdens0 = np.zeros(grid.nlens)
	xim = np.copy(xpos)
	xim[xim<0] = 0
	xim[xim>(width-1)] = 0
	yim = np.copy(ypos)
	yim[yim<0] = 0
	yim[yim>(width-1)] = 0
	sdens0 = data[np.round(xim), np.round(yim)]*scale	#Msol/pixel
	#------------------------------------------------------------------------#
	"""
	
	return grid, sdens0


def sdenspiemd(rr, grid, sigma):
	"""Return a matrix which associates a list of sigma^2 to an array of sdens
	in Msol/arcsec2
	rr=Dx2+ Dy2 where Dx=xi-xc Dy=yi-yc  (xc and yc are the centre of a clump)
	rc and rcut must be array of the same size in arcsec
	Sigma must be a scalar = 1.
	"""
	
	invGG = 1.02751e6	# valeur ajustee 'a la main avec un profil PIEMD'
	frr = np.copy(rr)
	for i in range(len(grid.rc) -1):
		frr[i,:] = sigma*sigma*grid.rcut[i]/2*invGG/(grid.rcut[i] - grid.rc[i])*(1./np.sqrt(grid.rc[i]**2 + rr[i,:]) - 1./np.sqrt(grid.rcut[i]**2 + rr[i,:]))
	return frr
	


def build_distmat(grid):
	"""From x and y, build a matrix of distances r2(i,j)"""

	xc = np.ones(grid.nlens)	# array |1 1 1 ... 1|
	xc = np.outer(xc, grid.x)	# matrix |x0 x1 x2 ... xN|
					#	 |x0 x1 x2 ... xN|
					#	 |	...	 |
					#	 |x0 x1 x2 ... xN|
	xc = xc - xc.T			# matrix |x0-x0 x1-x0 x2-x0 ... xN-x0|
					#	 |x0-x1 x1-x1 x2-x1 ... xN-x1|
					#	 |	      ...	     |
					#	 |x0-XN x1-xN x2-xN ... xN-xN|
	xc = xc*xc			# to the square

	yc = np.ones(grid.nlens)
	yc = np.outer(yc, grid.y)
	yc = yc - yc.T
	yc = yc*yc
	
	rr = xc + yc
	return rr



def get_pix2sec(hdr):
	xpos = [0, 1]
	ypos = [0, 0]
	w = wcs.WCS(hdr)
	xabs, yabs = w.all_pix2world(xpos, ypos, 1)
	dx = xabs[1] - xabs[0]
	dy = yabs[1] - yabs[0]
	pix2sec = np.sqrt(dx*dx + dy*dy)
	return pix2sec*3600.




def circle(grid):
	""" Overplot a circle"""
	t = np.arange(18.+1.)*20./180.*np.pi
	for i in range(len(grid.x)):
		plt.plot(grid.x[i]+grid.rc[i]*np.cos(t), grid.y[i]+grid.rc[i]*np.sin(t), 'k-')
	plt.plot(grid.x, grid.y, '+')


def multiscale_grid(grid, data):
	"""Build a multiscale grid with more resolution in high density regions.

	x, y (OUT) : position of the points in relative arcsec.
	Starting point and position of the 6 main triangles:
	     --------
	    / \ 1  / \
	   / 2 \  / 0 \
	  --------------
	   \ 3 /  \ 5 /
	    \ / 4  \ /
	     --------
	"""	
	
	scale = 1		# scaling kappa <-> absolute mass
	
	# Find the limits of the field in pixels	!!!!!!! PQ -2 !!!!!!!!!
	#------------------------------------------------------------------------#
	width = len(data[:,0])#-2
	height = len(data[0,:])#-2
	#------------------------------------------------------------------------#

	# Initial values for rc and rcut --> because we are sure that we go at
	# least to level1	
	#------------------------------------------------------------------------#
	rci = width/2.
	rcuti = rci*grid.ratio
	#------------------------------------------------------------------------#

	#Initial positions of the 7 first grid nodes
	#------------------------------------------------------------------------#
	x0=width/2.
	x1=x0+width/2.
	x2=x0+width/4.
	x3=x0-width/4.
	x4=x0-width/2.
	x5=x0-width/4.
	x6=x0+width/4.
	s3=np.sqrt(3.)/2.
	y0=height/2.
	y1=y0
	y2=y0+s3*width/2.
	y3=y0+s3*width/2.
	y4=y0
	y5=y0-s3*width/2.
	y6=y0-s3*width/2.
	#------------------------------------------------------------------------#

	# Add the 7 nodes to the grid
	#------------------------------------------------------------------------#
	grid.x[0:7] = [x0, x1, x2, x3, x4, x5, x6]
	grid.y[0:7] = [y0, y1, y2, y3, y4, y5, y6]
	grid.rc[0:7] = rci
	grid.rcut[0:7] = rcuti
	grid.nlens = 7
	#------------------------------------------------------------------------#

	# Divide triangle 0
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x1, x2], [y0, y1, y2], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x1, x2]), np.array([y0, y1, y2]), np.array([0,1,2]), rci, 0, 0)
	#------------------------------------------------------------------------#

	# Divide triangle 1
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x2, x3], [y0, y2, y3], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x2, x3]), np.array([y0, y2, y3]), np.array([0,2,3]), rci, 0, 0)
	#------------------------------------------------------------------------#

	# Divide triangle 2
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x3, x4], [y0, y3, y4], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x3, x4]), np.array([y0, y3, y4]), np.array([0,3,4]), rci, 0, 0)
	#------------------------------------------------------------------------#


	# Divide triangle 3
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x4, x5], [y0, y4, y5], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x4, x5]), np.array([y0, y4, y5]), np.array([0,4,5]), rci, 0, 0)
	#------------------------------------------------------------------------#


	# Divide triangle 4
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x5, x6], [y0, y5, y6], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x5, x6]), np.array([y0, y5, y6]), np.array([0,5,6]), rci, 0, 0)
	#------------------------------------------------------------------------#


	# Divide triangle 5
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0, x6, x1], [y0, y6, y1], data)*scale
	if (ssdenst > grid.threshold)or(grid.highlim > 0):
		grid = divide_tri(grid, data, np.array([x0, x6, x1]), np.array([y0, y6, y1]), np.array([0,6,1]), rci, 0, 0)
	#------------------------------------------------------------------------#

	return grid





def divide_tri(grid, data, xi, yi, idd, rci, leveli, noleft):
	r"""Divide a triangle into 4 subtriangles
	Append the new subtriangles positions to the x and y list
	Increment the nlens global variable
	
	           2
	         /   \
	        / [2] \
	       /       \
	    5 ---------- 4
	   /   \       /   \
	  / [0] \ [3] / [1] \
	 /       \   /       \
	0 -------  3 -------- 1
	
	 - xi : input triangle x corners positions in pixels
	 - yi : input triangle y corners positions in pixels
	 - id : indexes of the registered parents (size = 3),
	        Non registered = -1
	 - rci : scale radius of the input triangle
	 - leveli : level of the input triangle
	 - noleft : do not add point 5 to the X,Y list
	"""

       	# Scaling kappa <-> absolute mass
	#------------------------------------------------------------------------#
	scale=1
	#------------------------------------------------------------------------#


       	# Input triangle corners
	#------------------------------------------------------------------------#
        x0=xi[0]
        x1=xi[1]
        x2=xi[2]
        y0=yi[0]
        y1=yi[1]
        y2=yi[2]
	#------------------------------------------------------------------------#

        # Subtriangle corners and size
	#------------------------------------------------------------------------#
        x3=(x0+x1)/2.
        x4=(x1+x2)/2.
        x5=(x0+x2)/2.
        y3=(y0+y1)/2.
        y4=(y1+y2)/2.
        y5=(y0+y2)/2.
        rcs=rci/2.
        rcuts=rcs*grid.ratio
	#------------------------------------------------------------------------#

	# Reduce rc and rcut of the registered parent corners
	#------------------------------------------------------------------------#
	res = np.where(idd != -1)
	if len(res[0]) > 0:
		grid.rc[idd[res]] = rcs
		grid.rcut[idd[res]] = rcuts
	#------------------------------------------------------------------------#
	
	# Increment the grid arrays	
	#------------------------------------------------------------------------#
	grid.x[grid.nlens] = x4
	grid.y[grid.nlens] = y4
	grid.rc[grid.nlens] = rcs
	grid.rcut[grid.nlens] = rcuts
	ids = np.array([-1, grid.nlens, -1])		#corners 3, 4, 5
	grid.nlens+=1
	if not noleft:
		grid.x[grid.nlens] = x5
		grid.y[grid.nlens] = y5
		grid.rc[grid.nlens] = rcs
		grid.rcut[grid.nlens] = rcuts
		ids[2] = grid.nlens
		grid.nlens+=1
	#------------------------------------------------------------------------#

	t1 = time.time()

	# Max sdens in input image
	#------------------------------------------------------------------------#
	ssdenst = stat_sdens_tri([x0,x1,x2], [y0,y1,y2], data)*scale
	print ssdenst, grid.threshold
	#------------------------------------------------------------------------#



	# Recursive call to the function	
	#------------------------------------------------------------------------#
	if ((leveli+1 < grid.lowlim)&(ssdenst > grid.threshold))or(leveli+1 < grid.highlim):
		
		## Subtriangle 0
		#----------------------------------------------------------------#
		grid = divide_tri(grid, data, np.array([x0, x3, x5]), np.array([y0, y3, y5]), np.array([idd[0],-1,ids[2]]), rcs, leveli+1, noleft)
		#----------------------------------------------------------------#

		## Subtriangle 1
		#----------------------------------------------------------------#
		grid = divide_tri(grid, data, np.array([x3, x1, x4]), np.array([y3, y1, y4]), np.array([-1,idd[1],ids[1]]), rcs, leveli+1, 0)
		#----------------------------------------------------------------#

		## Subtriangle 2
		#----------------------------------------------------------------#
		grid = divide_tri(grid, data, np.array([x5, x4, x2]), np.array([y5, y4, y2]), np.array([ids[2],ids[1],idd[2]]), rcs, leveli+1, noleft)
		#----------------------------------------------------------------#

		## Subtriangle 3
		#----------------------------------------------------------------#
		grid = divide_tri(grid, data, np.array([x3, x4, x5]), np.array([y3, y4, y5]), ids, rcs, leveli+1, 1)
		#----------------------------------------------------------------#

	#------------------------------------------------------------------------#
	return grid


def stat_sdens_tri(xi, yi, data):
	"""Return the max value in the triangle"""



	# Find the limits of the field in pixels, no stats if less than 3x3 pixels
	#------------------------------------------------------------------------#
	width = len(data[:,0])
	height = len(data[0,:])
	if width*height < 9:
		return 0
	#------------------------------------------------------------------------#

	# Create matrix of abscisses (xmat) and ordonnees (ymat)
	#------------------------------------------------------------------------#
	xmat = np.outer(np.ones(width), np.arange(width))
	ymat = np.outer(np.ones(height), np.arange(height)).T
	#------------------------------------------------------------------------#

	# Create triangle mask
	#------------------------------------------------------------------------#
	mask = np.zeros((height, width))
	mask = inside(xi, yi, xmat, ymat, mask)	#Return mask=3 inside triangle
	#------------------------------------------------------------------------#

	# Take max value ine triangle
	#------------------------------------------------------------------------#
	res =np.where(mask == 3)
	msdent = np.max(data[res])
	return msdent
	#------------------------------------------------------------------------#



def determinant(xi, yi, xmat, ymat):
	"""Pour tout point M (defini par les matrices d'abscisses et d'ordonnees
	xmat et ymat), on calcule le produit vectoriel :
	M1M2XM1M = (xi[1]-xi[0], yi[1]-yi[0], 0)X(xM-xi[0], yM-yi[0], 0)
	ou l'input est 2 coins du triangle M1=(xi[0],yi[0]) et M2=(xi[1],yi[1])
	"""


	dx = xi[1] - xi[0]
	dy = yi[0] - yi[1]
	#dxy1 = np.multiply(dx, ymat)
	#dxy2 = np.multiply(dy, xmat)

	
	#det = np.add(np.add(dxy1, dxy2), xi[0]*yi[1] - yi[0]*xi[1])
	det = np.add(np.add(np.multiply(dx, ymat), np.multiply(dy, xmat)), xi[0]*yi[1] - yi[0]*xi[1])


	#det = xi[0]*(yi[1] - ymat) - yi[0]*(xi[1] - xmat) + xi[1]*ymat - yi[1]*xmat
	#det = xi[0]*yi[1] - yi[0]*xi[1] + (xi[1] - xi[0])*ymat + (yi[0] - yi[1])*xmat
	#det = xi[0]*yi[1] - yi[0]*xi[1] + dx*ymat + dy*xmat
	return det


def inside(xi, yi, xmat, ymat, mask):
	"""Return mask=3 inside the triangle defined by the xi and yi triplets of points.

	xmat (IN) : squared matrix of abscissa
	ymat (IN) : squared matrix of ordinates
	mask (OUT) : must be a squared matrix initialised to 0.
	mask, xmat and ymat must have the same size
	The idea is to check for each side of triangle if points are on the "good" side by comparing the sign of cross-products
	"""
	
	d = np.linalg.det(np.array([xi, yi, [1, 1, 1]]))	# Equivalent au produit vectoriel de 2 cote du triangle
	
	s = determinant([xi[0], xi[1]], [yi[0], yi[1]], xmat, ymat)*d
	mask[np.where(s > 0.)]+=1
	
	s = determinant([xi[1],xi[2]], [yi[1], yi[2]], xmat, ymat)*d
	mask[np.where(s > 0.)]+=1
	
	s = determinant([xi[2],xi[0]], [yi[2],yi[0]], xmat, ymat)*d
	mask[np.where(s > 0.)]+=1

	return mask




class Grid:
	"""Class defining a grid"""

	def __init__(self, nlens):
		"""Class constructeur"""
		self.nlens = nlens
		self.x = np.zeros(nlens)
		self.y = np.zeros(nlens)
                self.rc = np.zeros(nlens)
                self.rcut = np.zeros(nlens)
		self.threshold = 0.
		self.highlim = 0.




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Create multiscale potential grid from density map')
	parser.add_argument('fits_map', 
			    help='Fits file with density map, input to create the grid')
	parser.add_argument('threshold', type=float, 
			    help='Tthreshold on the density map to split triangles')
	parser.add_argument('-ll', action='store', default=7, dest='lowlim', 
			    type=float, help='Maximum number of triangle split, default=7')
	parser.add_argument('-hl', action='store', dest='highlim', default=1, 
			    type=float, help='Minimum number of triangle split, default=1')
	parser.add_argument('-r', action='store', dest='ratio', default=3, 
			    type=float, help='Ratio rcut/rc, default=3')
	parser.add_argument('--reg', dest='regular', action='store_true', default=False, help='Creates regular grid, with lowlim triangle splits, default=False')
	args = parser.parse_args()

	# On redefini les parametres d'input
	#------------------------------------------------------------------------#
	fits_map = args.fits_map
	threshold = args.threshold
	lowlim = args.lowlim
	highlim = args.highlim
	ratio = args.ratio
	regular = args.regular
	#------------------------------------------------------------------------#

	t0 = time.time()
	main(fits_map, threshold, lowlim, highlim, ratio, regular)


