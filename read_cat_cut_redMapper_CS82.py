import numpy as np
from astropy.io import fits
#import cosmolopy.distance as cd
from scipy import spatial
import time
from astropy.cosmology import LambdaCDM
import sys
#~
#cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.7}
cosmo  = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

############ Group Data ###########


grupos = fits.open('/mnt/clemente/lensing/KiDS/redmapper_dr8_public_v6.3_catalog.fits')

name='redMapper_CS82_rmax30'

alfa0  = grupos[1].data['RA']
delta0 = grupos[1].data['DEC']
alfa0[alfa0>275.]=alfa0[alfa0>275.]-360.

#mgroup = (delta0>-1.)*(delta0<1.)*(alfa0<50.)*(alfa0>-50.)

#alfa0  = alfa0[mgroup]
#delta0 = delta0[mgroup]

zl     = grupos[1].data['Z_LAMBDA']#[mgroup]
#grupos = grupos[1].data[mgroup]

print '######### LENS DISTANCES ###########'
#dl = cd.angular_diameter_distance(zl, z0=0, **cosmo)
dl = np.array(cosmo.angular_diameter_distance(zl))

kpcscale=dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0

MAX_MPC = 30.
	
R_grad = (MAX_MPC*1000./kpcscale)/3600.

############ Lensfit Cat ###########

hdulist = fits.open('/mnt/clemente/lensing/cs82_cat/cs82_combined_lensfit_sources_nozcuts_aug_2015.fits')
tbdata=hdulist[1].data
hdulist.close()

alpha_J2000 = tbdata.field('ALPHA_J2000')
delta_J2000 = tbdata.field('DELTA_J2000')
X_IMAGE     = tbdata.field('X_IMAGE')
Y_IMAGE     = tbdata.field('Y_IMAGE')

SE_ID       = tbdata.field('SE_ID')
IM_NUM      = tbdata.field('IM_NUM')

MAG_AUTO    = tbdata.field('MAG_AUTO')

zb          = tbdata.field('BPZ_ZPHOT')
BPZ_LOW95   = tbdata.field('BPZ_LOW95')
BPZ_ODDS    = tbdata.field('BPZ_ODDS')
E1          = tbdata.field('E1CORR')
E2          = tbdata.field('E2CORR')
WEIGHT      = tbdata.field('WEIGHT')
M           = tbdata.field('M')


lenscat = np.array([SE_ID,IM_NUM,alpha_J2000,delta_J2000,X_IMAGE,Y_IMAGE,MAG_AUTO,zb,BPZ_LOW95,BPZ_ODDS,E1,E2,M,WEIGHT]).T

mincut=delta_J2000.min()
maxcut=mincut+0.26

for j in np.arange(10):
	print mincut,maxcut
	
	mask1=(delta_J2000>mincut)*(delta_J2000<maxcut)
	
	print 'Correlacionando parte ',j+1,' ...'
	
	t1=time.time()
		
		
	print '######### CORRELACIONANDO CATALOGOS ##########'
		
	tree = spatial.cKDTree(np.array([alfa0,delta0]).T)
	
	dist,ind = tree.query(np.array([alpha_J2000[mask1],delta_J2000[mask1]]).T,k=150)
	
	print '######### MASCARAS ##########'
	
	ZB = np.array([zb[mask1],]*150).T
	
	mask = (dist < R_grad[ind])*(ZB > zl[ind]) # mascara de galaxias dentro del radio
	
	del(dist)
		
	ind_g = ind[mask] # arreglo unidemensional de los ids de grupos de cada galaxia 
	sum_mask = np.sum(mask,axis=1) #suma de la mascara True=1, false=0, equivale a cantidad de veces que cada galaxia pertenece a un grupo como background
	
	print 'mas de 150', (sum_mask==150.).sum()
	
	print '#------- parameters of the lensfit cat'
	
	lcat = np.repeat(lenscat[mask1,:], sum_mask, axis = 0)
	Zs = np.repeat(zb[mask1], sum_mask)
	#DS = cd.angular_diameter_distance(Zs, z0=0, **cosmo)
	DS = np.array(cosmo.angular_diameter_distance(Zs))

	
	print '#---------- parameters of group cat'
	
	#gcat = grupos[ind_g]
	gcat = grupos[1].data[ind_g]
	DL = dl[ind_g]
	Zl = zl[ind_g]
	
	del(ind_g)
	
	print '#-------------- computing distances'
	

	DLS=np.array(cosmo.angular_diameter_distance_z1z2(Zl,Zs))
	#DLS=np.zeros(sum_mask.sum())
	
	del(sum_mask)	
	

	print '-----------------------------------------------------------------'
	print '                 MAKING THE CATALOGUE FOR GALAXIES               '
	print '================================================================='
	


		
	tbhdu = fits.BinTableHDU.from_columns(
			[fits.Column(name='SE_ID', format='E', array=lcat[:,0]),
			fits.Column(name='IM_NUM', format='D', array=lcat[:,1]),
			fits.Column(name='RAJ2000', format='D', array=lcat[:,2]),
			fits.Column(name='DECJ2000', format='D', array=lcat[:,3]),
			fits.Column(name='X_IMAGE', format='E', array=lcat[:,4]),
			fits.Column(name='Y_IMAGE', format='E', array=lcat[:,5]),
			fits.Column(name='MAG_AUTO', format='E', array=lcat[:,6]),
			fits.Column(name='Z_B', format='E', array=lcat[:,7]),
			fits.Column(name='BPZ_LOW95', format='E', array=lcat[:,8]),
			fits.Column(name='ODDS', format='E', array=lcat[:,9]),
			fits.Column(name='E1', format='E', array=lcat[:,10]),
			fits.Column(name='E2', format='E', array=lcat[:,11]),
			fits.Column(name='M', format='E', array=lcat[:,12]),
			fits.Column(name='WEIGHT', format='E', array=lcat[:,13]),
			fits.Column(name='ID_c', format='26A', array=gcat['ID']),
			fits.Column(name='RA', format='D', array=gcat['RA']),
			fits.Column(name='DEC', format='D', array=gcat['DEC']),
			fits.Column(name='z',format='E', array=gcat['Z_LAMBDA']),
			fits.Column(name='zerr',format='E', array=gcat['Z_LAMBDA_ERR']),
			fits.Column(name='lambda', format='D', array=gcat['LAMBDA']),
			fits.Column(name='lambda_err',format='D', array=gcat['LAMBDA_ERR']),
			fits.Column(name='DS',format='E', array=DS),
			fits.Column(name='DLS',format='E', array=DLS),
			fits.Column(name='DL',format='E', array=DL)])
		
		
		
			
			
	tbhdu.writeto('gx_'+name+'_'+str(int(j))+'.fits')
#tbhdu.writeto('gx_'+name+'.fits')

	mincut=maxcut
	maxcut=mincut+0.26
	
	del(tbhdu)
	
	t2=time.time()
	print 'listo ',j+1
	print 'time', t2-t1
	


t0 = fits.open('gx_'+name+'_0.fits')
t1 = fits.open('gx_'+name+'_1.fits')
t2 = fits.open('gx_'+name+'_2.fits')
t3 = fits.open('gx_'+name+'_3.fits')
t4 = fits.open('gx_'+name+'_4.fits')
t5 = fits.open('gx_'+name+'_5.fits')
t6 = fits.open('gx_'+name+'_6.fits')
t7 = fits.open('gx_'+name+'_7.fits')
t8 = fits.open('gx_'+name+'_8.fits')
t9 = fits.open('gx_'+name+'_9.fits')


nrows0 = t0[1].data.shape[0]
nrows1 = t1[1].data.shape[0]
nrows2 = t2[1].data.shape[0]
nrows3 = t3[1].data.shape[0]
nrows4 = t4[1].data.shape[0]
nrows5 = t5[1].data.shape[0]
nrows6 = t6[1].data.shape[0]
nrows7 = t7[1].data.shape[0]
nrows8 = t8[1].data.shape[0]
nrows9 = t9[1].data.shape[0]

nrows = nrows0 + nrows1 + nrows2 + nrows3 + nrows4 + nrows5 + nrows6 + nrows7 + nrows8 + nrows9

hdu = fits.BinTableHDU.from_columns(t0[1].columns, nrows=nrows)
for colname in t0[1].columns.names:
	limsup=nrows0+nrows1
	print colname
	hdu.data[colname][nrows0:limsup] = t1[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows2
	hdu.data[colname][liminf:limsup] = t2[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows3
	hdu.data[colname][liminf:limsup] = t3[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows4
	hdu.data[colname][liminf:limsup] = t4[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows5	
	hdu.data[colname][liminf:limsup] = t5[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows6
	hdu.data[colname][liminf:limsup] = t6[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows7
	hdu.data[colname][liminf:limsup] = t7[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows8	
	hdu.data[colname][liminf:limsup] = t8[1].data[colname]
	liminf=limsup
	limsup=limsup+nrows9
	hdu.data[colname][liminf:limsup] = t9[1].data[colname]
    
hdu.writeto('gx_'+name+'.fits')   
	

