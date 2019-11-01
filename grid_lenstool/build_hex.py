#!/usr/local/anaconda/bin/python
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from astropy import wcs
from astropy.io import fits

def quit():
    sys.exit()

def main(sdens,cmcat,zl,srccat,mask):
	#------------------------------------------------------------------------#
	outfile = 'hex.par'


	print "Params = {} {} {} {} \n".format(sdens, cmcat, zl, srccat)


	# Read reference RA Dec from sdens file
        #------------------------------------------------------------------------#
	f = open(sdens, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
                if line[0] == "#":
                        ra = float(line.split()[2])
                        dec = float(line.split()[3])
                        print ra, dec
        #------------------------------------------------------------------------#


	# Read the catalogue of cluster members
        #------------------------------------------------------------------------#
        nlens = 0
	
	if os.path.isfile(cmcat):
		f = open(cmcat)
		lines = f.readlines()
		f.close()
		for line in lines:
			if line[0] != '#':
				nlens+=1
	else:
		print 'Error: file {} not found \n'.format(cmcat)
		quit() 
	print '{} galaxies in {}'.format(nlens, cmcat)
	#------------------------------------------------------------------------#


	# Read the catalogue of positions, rc, rcut for all the clumps
        #------------------------------------------------------------------------#
	nmsgrid = nlens
	ntot = nlens
	if os.path.isfile(sdens):
		f = open(sdens)
		lines = f.readlines()
		f.close()
		i = 0
		x = []
		y = []
		idd = []
		rc = []
		rcut = []
		vdisp = []
		if mask:
			masked = []
		for line in lines:
			if line[0] == '#': continue
			ntot+=1
			i+=1
			idd.append(line.split()[0])
			x.append(float(line.split()[1]))
			y.append(float(line.split()[2]))
			rc.append(float(line.split()[3]))
			rcut.append(float(line.split()[4]))
			vdisp.append(float(line.split()[5]))
			if mask:
				masked.append(float(line.split()[6]))
				if float(line.split()[6])  == 0:
					nlens+=1
			else:
				nlens+=1
		print '{} RBFs found in {} \n'.format(i, sdens)
		xmin = min(x)
		xmax = max(x)
		ymin = min(y)
		ymax = max(y)
		dx = -(xmax - xmin)*3600.*np.cos(dec)
		dy = (ymax - ymin)*3600.
		print 'Min max in {} ({}, {}, {}, {}) \n dx, dy = ({}, {})\n'.format(sdens, xmin, xmax, ymin, ymax, dx, dy)
	else:
		print 'Error: file {} not found \n'.format(sdens)
		quit()
	print '{} total clumps read \n'.format(nlens)
	#------------------------------------------------------------------------#


	# Compute K-correction for potfile
	#------------------------------------------------------------------------#
	kcorr = 1.689*zl*zl*zl - 3.736*zl*zl + 2.339*zl - 0.9524	# From LePhare CFHT_K, CWW_Ell, LCDM(0.27,0.73,h=0.70). Good to 1.5% between z=0.3 and 1.
	H0 = 70.
	Mstar = -23.85 + 5.*np.log10(H0/100.)
	D = 1547.*zl*zl + 5239*zl - 99.7	 # LCDM(0.27, 0.73, h=0.70) good to about 3% btw z=0.1 and 1.4
	mstar = Mstar - kcorr + 5.*np.log10(D) + 25.
	smstar = '{:.2f}'.format(mstar)
	#------------------------------------------------------------------------#


	# Write Lenstool input file
	#------------------------------------------------------------------------#
	fout = open(outfile, 'w')
	fout.write("runmode \n\
        reference 3 {} {} \n\
        inverse   3 0.1 100 \n\
        mass 3 200 {} mass.fits \n\
        end\n".format(ra, dec, zl))
	fout.write("image \n\
        arcletstat 8 1 {} \n\
        sigell 0.27 \n\
        end\n".format(srccat))	
	fout.write("source \n\
	grid 0 \n\
        n_source 0 \n\
        dist_z 2 0.44 2.94 \n\
        smail 2. 1.3 1.9 \n\
        elip_max 0.25 \n\
        end \n")   
	fout.write("grille \n\
        nombre      128 \n\
        nlentille   {} \n\
        nmsgrid   {} \n\
        end\n".format(nlens, nmsgrid))
	fout.write("potfile \n\
        filein 3 {} \n\
        type 81 \n\
        corekpc 0.2 \n\
        mag0    19.76  # in CFHT_K for CWW_Ell \n\
        sigma  0 119 185 \n\
        cutkpc  0 85 50 \n\
        zlens {} \n\
        end\n".format(cmcat, zl))

	l = 0	
	for i in range(ntot - nmsgrid):
		if mask:
			if masked[i]:
				continue
		fout.write("potential {} \n".format(l+3))
		fout.write("        profil 81 \n")
		fout.write("        x_centre  {} \n".format(x[i]))
		fout.write("        y_centre  {} \n".format(y[i]))
		fout.write("        core_radius  {} \n".format(rc[i]))
		fout.write("        cut_radius  {} \n".format(rcut[i]))
		fout.write("        v_disp  {} \n".format(vdisp[i]))
		fout.write("        z_lens  {} \n".format(zl))
		fout.write("        end \n")
		l+=1

	fout.write("cosmologie \n\
        H0         70.000 \n\
        omegaM     0.270 \n\
        omegaX     0.730 \n\
        omegaK     0. \n\
        wX         -1.000 \n\
        end \n")
        fout.write("champ \n\
        xmin     -100.000 \n\
        xmax     100.000 \n\
        ymin     -100.000 \n\
        ymax     100.000 \n\
        dmax   160 \n\
        end \n")
        fout.write("fini")
	fout.close()	
	#------------------------------------------------------------------------#




if __name__ == '__main__':
        parser = argparse.ArgumentParser(description = 'Create Lenstool input file in grid mode')
        parser.add_argument('sdens',
                            help='Catalogue of grid potentials, built with fit_hex.py')
        parser.add_argument('cmcat',
                            help='Galaxy catalogue')
        parser.add_argument('zl', type=float,
                            help='Lens redshift')
        parser.add_argument('srccat',
                            help='Source catalogue')
        parser.add_argument('--mask', dest='mask', action='store_true', 
			    help='Define if there is a mask column in sdens to show the SL region')
        args = parser.parse_args()
	
		        # On redefini les parametres d'input
        #------------------------------------------------------------------------#
        sdens = args.sdens
	cmcat = args.cmcat
	zl = args.zl
	srccat = args.srccat
	mask = args.mask
	
	main(sdens,cmcat,zl,srccat,mask)
