import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def magfits(ra,dec,mag,npix,out_name):
	
	
	escx=ra.max()-ra.min()
	escy=dec.max()-dec.min()

	B=np.zeros((npix,npix),float)

	stepx=escx/float(npix)
	stepy=escy/float(npix)

	crpix1 = (npix-1)/2. + 0.5	
	crpix2 = (npix-1)/2. + 0.5

	cdelt1 = -1.*stepx
	cdelt2 = stepy
	
	crval1 = ra.min()+escx/2.
	crval2 = dec.min()+escy/2.
	
	hdu = fits.PrimaryHDU(B)
	hdu.header['BZERO'] = (0.,'BZERO')
	hdu.header['BSCALE'] = (1.,'BSCALE')
	hdu.header['EQUINOX'] = (2000.,'EQUINOX')
	hdu.header['RADECSYS'] = ('FK5    ','RADECSYS')
	hdu.header['CTYPE1'] = ('RA---TAN','CTYPE1')
	hdu.header['CRVAL1'] = (crval1,'CRVAL1')
	hdu.header['CRPIX1'] = (crpix1,'CRPIX1')
	hdu.header['CDELT1'] = (cdelt1,'CDELT1')
	hdu.header['CUNIT1'] = ('deg   ','CUNIT1')
	hdu.header['CTYPE2'] = ('DEC--TAN','CTYPE2')
	hdu.header['CRVAL2'] = (crval2,'CRVAL2')
	hdu.header['CRPIX2'] = (crpix2,'CRPIX2')
	hdu.header['CDELT2'] = (cdelt2,'CDELT2')
	hdu.header['CUNIT2'] = ('deg   ','CUNIT2')
	
	w = WCS(hdu.header)
	#plt.plot(ra,dec,'k.')
	for m in range(npix):
		for n in range(npix):
			
			lim    = w.all_pix2world(np.array([m+1.5,m+2.5]),np.array([n+1.5,n+2.5]),2)
			ramin  = lim[0][1]
			ramax  = lim[0][0]
			decmin = lim[1][0]
			decmax = lim[1][1]

			#plt.plot([ramin,ramax],[decmin,decmax],'bo')	
			for k in range(len(ra)):
				
				if (ramin<ra[k]<ramax) and (decmin<dec[k]<decmax):
					#plt.plot([ramin,ramax],[decmin,decmax],'ro')	
					B[n,m]= B[n,m]+10.**(-0.4*mag[k])

	
	hdu.writeto(out_name,overwrite=True)


