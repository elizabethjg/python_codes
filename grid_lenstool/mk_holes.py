#!/usr/local/anaconda/bin/python
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from astropy.table import Table
from astropy import wcs
from astropy.io import fits

def main(sdens,region):
	outfile = 'sdens_holes.dat'
		# Read RA_ref, Dec_ref from sdens file
	#------------------------------------------------------------------------#
	f = open(sdens, 'r')
	lines = f.readlines()
	f.close()
	for line in lines:
		if line[0] == "#":
			ra_ref = float(line.split()[2])
			dec_ref = float(line.split()[3])
			print ra_ref, dec_ref
	#------------------------------------------------------------------------#


	# Read input file
	#------------------------------------------------------------------------#
	tin = Table.read(sdens, format='ascii.no_header', comment='#', names=['idd', 'ra_rel', 'dec_rel', 'rc', 'rcut', 'vdisp'])
	n = len(tin['idd'])
	print 'Initial number of WL sources = {} \n'.format(n)
	#------------------------------------------------------------------------#


	# Read region file
	#------------------------------------------------------------------------#
	fregion = open(region, 'r')
	lines = fregion.readlines()
	fregion.close()
	edgex = []
	edgey = []
	for line in lines:
		if line.split('(')[0] == 'polygon':
			points = line.split('(')[1][:-2].split(',')
			for j in range(len(points)/2):
				edgex.append(float(points[2*j]))
				edgey.append(float(points[2*j+1]))
	#------------------------------------------------------------------------#


	ok = np.zeros(n)
	# Convert to absolute positions
	#------------------------------------------------------------------------#
	ra, dec = rel2abs(tin['ra_rel'], tin['dec_rel'], ra_ref, dec_ref)
	#------------------------------------------------------------------------#

	# Check if in region
	#------------------------------------------------------------------------#
	ok = inpoly2(ra, dec, edgex, edgey)
	#------------------------------------------------------------------------#


	# Add column to table and write
	#------------------------------------------------------------------------#
	tin['mask'] = ok
	tin.write(outfile, format='ascii.no_header', comment='#', overwrite=True)
	#------------------------------------------------------------------------#



	# Print results
	#------------------------------------------------------------------------#
	select = ok[np.where( ok == 1)]
	print 'Number of WL galaxies in the hole = {} \n'.format(len(select))
	print 'New number of WL galaxies = {} \n'.format(n - len(select))
	#------------------------------------------------------------------------#

def rel2abs(ra_rel, dec_rel, ra_ref, dec_ref):
	''' Convert relative coordinates to absolute '''

	dec = dec_rel/3600. + dec_ref
	ra = ra_ref - ra_rel/(3600.*np.cos(dec*np.pi/180.))
	return [ra, dec]



def inpoly2(ra, dec, edgex, edgey):
	'''    
	Determines if a point P(px, py) in inside or outside a polygon.
	The method used is the ray intersection method based on the
	Jordan curve theorem. Basically, we trace a "ray" (a semi line)
	from the point P and we calculate the number of edges that it
	intersects. If this number is odd, P is inside.  '''
	
	ok = []
	N = len(edgex)		# Number of vertices
	nc = 0			# Number of edge crossings

	# Test input
	#------------------------------------------------------------------------#
	if N < 3:
		print 'A polygon must have at least three vertices'
		quit()
	if len(edgex) != len(edgey):
		print 'Polygon must have same number of X and Y coordinates'
		quit()
	#------------------------------------------------------------------------#

	# For each point P in list
	#------------------------------------------------------------------------#
	k = 0
	for x,y in zip(ra,dec):
		nc = 0
		xv = np.copy(edgex)
		yv = np.copy(edgey)
		# Change coordinate system : place P at centre of the coordinate system
		#----------------------------------------------------------------#
		xv = xv - x
		yv = yv - y
		#----------------------------------------------------------------#
		# Calculate crossing
		## The positive half of the x axis is chosen as the ray
		## We must determine how many edges cross the x axis with x>0
		#----------------------------------------------------------------#
		for i in range(N): 
			Ax = xv[i]	# First vertice of edge
			Ay = yv[i]

			if (i == N-1):
				Bx = xv[0]
				By = yv[0]
			else:
				Bx = xv[i+1]	# Second vertice of edge
				By = yv[i+1]
			
			# We define two regions in the plan: R1/ y<0 and R2/ y>=0.
			# Note that the case y=0 (vertice on the ray) is included in R2.
			if Ay < 0:
				signA = -1
			else:
				signA = +1
			if By < 0:
				signB = -1
			else:
				signB = +1

			# The edge crosses the ray only if A and B are in different regions.
			# If a vertice is only the ray, the number of crossings will still
			# be correct.
			if (signA*signB < 0):
				
				# If Ax>0 and Bx>0 then the edge crosses the postitive x axis
				if ((Ax > 0)&(Bx > 0)):
					nc = nc+1
				# Otherwise (end points are in diagonally opposite quadrants)
				# we must calculate the intersection
				else:
					x = Ax-(Ay*(Bx-Ax))/(By-Ay)
					if (x > 0):
						nc = nc+1
		# If inside then uneven
		# If outside then even
		nc = nc%2
		ok.append(nc)
		#----------------------------------------------------------------#
	ok = np.array(ok)
	return ok
	#------------------------------------------------------------------------#




if __name__ == '__main__':
        parser = argparse.ArgumentParser(description = 'Add mask column to sdens catalogue corresponding to region from DS9, output = sdens_holes.dat')
        parser.add_argument('sdens',
                            help='Catalogue of grid potentials, built with fit_hex.py')
        parser.add_argument('region',
                            help='DS9 region file to filter')
        args = parser.parse_args()
	

	# Redefine input parameters
	#------------------------------------------------------------------------#
	sdens = args.sdens
	region = args.region
	#------------------------------------------------------------------------#
	
	main(sdens,region)
