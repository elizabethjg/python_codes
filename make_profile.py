import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from profiles_fit import *
from maria_func import *
from astropy.io import fits
import pylab as pylab
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad
from matplotlib import gridspec
from multiprocessing import Pool
from multiprocessing import Process
import time

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

def covariance_matrix(array):
	
	nobs    = len(array)
	
	CV = np.zeros((nobs,nobs))
	
	for i in range(nobs):
		nsample = len(array[i])
		mean_ei = np.mean(array[i])
		
		for j in range(nobs):

			mean_ej = np.mean(array[j]) 
			
			CV[i,j] = np.sqrt((abs(array[i] - mean_ei)*abs(array[j] - mean_ej)).sum()/nsample)/np.sqrt(array[i].std()*array[j].std())
	
	return CV

def bootstrap_errors_stack(et,ex,peso,nboot,array):
	unique = np.unique(array)
	with NumpyRNGContext(1):
		bootresult = bootstrap(unique, nboot)
		
	et_means = np.array([np.average(et[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in bootresult])
	ex_means = np.array([np.average(ex[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in bootresult])
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means

def errors_disp_halos(et,ex,peso,array):
	unique = np.unique(array)
		
	et_means = np.array([np.average(et[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in unique])
	ex_means = np.array([np.average(ex[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in unique])
	
	return np.std(et_means)/np.sqrt(len(unique)),np.std(ex_means)/np.sqrt(len(unique)),et_means,ex_means

def bootstrap_errors(et,ex,peso,nboot):
	index=np.arange(len(et))
	with NumpyRNGContext(1):
		bootresult = bootstrap(index, nboot)
	INDEX=bootresult.astype(int)
	ET=et[INDEX]	
	EX=ex[INDEX]	
	W=peso[INDEX]	
	
	et_means=np.average(ET,axis=1,weights=W)
	ex_means=np.average(EX,axis=1,weights=W)
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means

def qbootstrap_errors(et,ex,peso,angle,nboot):
	index=np.arange(len(et))
	with NumpyRNGContext(1):
		bootresult = bootstrap(index, nboot)
	INDEX=bootresult.astype(int)
	ET=et[INDEX]	
	EX=ex[INDEX]	
	W=peso[INDEX]	
	A = angle[INDEX]
	
	et_means = np.sum((ET*np.cos(A)*W),axis=1)/np.sum(((np.cos(A)**2)*W),axis=1)
	ex_means = np.sum((EX*np.sin(A)*W),axis=1)/np.sum(((np.sin(A)**2)*W),axis=1)
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means


def shear_profile_log(RIN,ROUT,r,et,ex,peso,m,sigma_c,
                      ndots=15,stepbin=False,booterror_flag=False,
                      lin=False,boot_stack=[],nboot = 100,cov_matrix = False):
	
	'''
	COMPUTE DENSITY PROFILE
	
	------------------------------------------------------
	INPUT
	------------------------------------------------------
	RIN               (float) Radius in kpc from which it is going 
	                  to start binning
	ROUT              (float) Radius in kpc from which it is going 
	                  to finish binning
	r                 (float array) distance from the centre in kpc
	et                (float array) tangential ellipticity component
	                  scaled by the critical density (M_sun/pc^2)
	ex                (float array) cross ellipticity component
	                  scaled by the critical density (M_sun/pc^2)
	peso              (float array) weight for each shear component
	                  scaled according to sigma_c^-2
	m                 (float array) correction factor
	sigma_c           (float array) critical density (M_sun/pc^2)
	                  used only to compute the error in each bin
	                  considering shape noise only
	ndots             (int) number of bins in the profile
	stepbin           (float) length of the bin instead of ndots
	                  if False, it is going to use ndots
	booterror_flag    (bool) if True it is going to use bootstraping
	                  to compute the error in each bin
	lin               (bool) if True it is going to use linalg spacing
	                  between the bins
	boot_stack        (array) used to do the bootstraping, if it is empty
	                  the bootstrap is going to be executed over the whole
	                  sample
	nboot             (int) number of bootstrap repetitions
	cov_matrix        (bool) if true it is going to compute the covariance matrix
	                  - this is still in testing process

	------------------------------------------------------
	OUTPUT
	------------------------------------------------------


	'''


	
		
	if lin:
		if stepbin:
			nbin = int((ROUT - RIN)/stepbin)
		else:
			nbin = int(ndots)
		bines = np.linspace(RIN,ROUT,num=nbin+1)
	else:
		if stepbin:
			nbin = int((np.log10(ROUT) - np.log10(RIN))/stepbin)
		else:
			nbin = int(ndots)
		bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=nbin+1)
		
	if cov_matrix and len(boot_stack):

		ides       = np.unique(boot_stack)
		digit      = np.digitize(r,bines)
		totbines   = np.arange(1,nbin+1)
		maskid     = np.array([all(np.in1d(totbines,digit[boot_stack==x])==True) for x in ides])
		maskides   = np.in1d(boot_stack,ides[maskid])
		boot_stack = boot_stack[maskides]
		r          = r[maskides]
		et         = et[maskides]
		ex         = ex[maskides]
		peso       = peso[maskides]
		m          = m[maskides]
		sigma_c    = sigma_c[maskides]
		if nboot > maskid.sum() and len(boot_stack):
			nboot = maskid.sum()


	etboot = []
	exboot = []


	SHEAR=np.zeros(nbin,float)
	CERO=np.zeros(nbin,float)
	R=np.zeros(nbin,float)
	err=np.zeros(nbin,float)
	error_et=np.zeros(nbin,float)
	error_ex=np.zeros(nbin,float)
	Mcorr=np.zeros(nbin,float)
	N=np.zeros(nbin,float)
		
	
	for BIN in np.arange(nbin):
		# print 'BIN',BIN
		rin  = bines[BIN]
		rout = bines[BIN+1]
		maskr=(r>=rin)*(r<rout)	
		w2=peso[maskr]
		pes2=w2.sum()			
		shear=et[maskr]
		cero=ex[maskr]
		ERR=((sigma_c[maskr]*w2)**2)
		mcorr=m[maskr]
		n=len(shear)
		R[BIN]=rin+(rout-rin)/2.0	
		#~print n
		N[BIN] = n
		if n == 0:
			SHEAR[BIN]=0.0
			CERO[BIN]=0.0
			err[BIN]=0.
			error_et[BIN],error_ex[BIN]=0.,0.
			Mcorr[BIN]=1.
		else:	
			SHEAR[BIN]=np.average(shear,weights=w2)
			CERO[BIN]=np.average(cero,weights=w2)
			sigma_e=(0.28**2.)
			ERR2=(ERR*sigma_e).sum()
			err[BIN]=((ERR2)/((pes2.sum())**2))**0.5			
			Mcorr[BIN]=1+np.average(mcorr,weights=w2)
			if booterror_flag:
				if len(boot_stack):
					error_et[BIN],error_ex[BIN],etboot0,exboot0 = bootstrap_errors_stack(shear,cero,w2,nboot,boot_stack[maskr])
				else:
					error_et[BIN],error_ex[BIN],etboot0,exboot0 = bootstrap_errors(shear,cero,w2,nboot)
				etboot += [etboot0]
				exboot += [exboot0]
				#SHEAR[BIN]=np.average(etboot0)
				#SHEAR[BIN]=np.average(exboot0)
			elif len(boot_stack):
				error_et[BIN],error_ex[BIN],etboot0,exboot0 = errors_disp_halos(shear,cero,w2,boot_stack[maskr])
				etboot += [etboot0]
				exboot += [exboot0]
			else:
				error_et[BIN],error_ex[BIN] = err[BIN],err[BIN]
		
	if cov_matrix:
		CVet = covariance_matrix(etboot)
		CVex = covariance_matrix(exboot)
	else:
		CVet = None
		CVex = None
		
	return [R,SHEAR/Mcorr,CERO/Mcorr,err/Mcorr,nbin,error_et/Mcorr,error_ex/Mcorr,N,CVet,CVex]


def quadrupole_profile_log(RIN,ROUT,r,et,ex,peso,m,sigma_c,angle,
                      ndots=15,stepbin=False,booterror_flag=False,
                      lin=False,nboot = 100):
		
	if lin:
		if stepbin:
			nbin = int((ROUT - RIN)/stepbin)
		else:
			nbin = int(ndots)
		bines = np.linspace(RIN,ROUT,num=nbin+1)
	else:
		if stepbin:
			nbin = int((np.log10(ROUT) - np.log10(RIN))/stepbin)
		else:
			nbin = int(ndots)
		bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=nbin+1)
		

	etboot = []
	exboot = []


	SHEAR=np.zeros(nbin,float)
	CERO=np.zeros(nbin,float)
	R=np.zeros(nbin,float)
	err=np.zeros(nbin,float)
	error_et=np.zeros(nbin,float)
	error_ex=np.zeros(nbin,float)
	Mcorr=np.zeros(nbin,float)
	N=np.zeros(nbin,float)
		
	
	for BIN in np.arange(nbin):
		# print 'BIN',BIN
		rin    = bines[BIN]
		rout   = bines[BIN+1]
		maskr  = (r>=rin)*(r<rout)	
		w2     = peso[maskr]
		pes2   = w2.sum()			
		shear  = et[maskr]
		cero   = ex[maskr]
		ERR    = ((sigma_c[maskr]*w2)**2)
		mcorr  = m[maskr]
		n      = len(shear)
		R[BIN] = rin+(rout-rin)/2.0	
		#~print n
		N[BIN] = n
		if n == 0:
			SHEAR[BIN]=0.0
			CERO[BIN]=0.0
			err[BIN]=0.
			error_et[BIN],error_ex[BIN]=0.,0.
			Mcorr[BIN]=1.
		else:	
			SHEAR[BIN]=np.sum(shear*np.cos(angle[maskr])*w2)/np.sum((np.cos(angle[maskr])**2)*w2)
			CERO[BIN]=np.sum(cero*np.sin(angle[maskr])*w2)/np.sum((np.sin(angle[maskr])**2)*w2)
			sigma_e=(0.28**2.)
			ERR2=(ERR*sigma_e).sum()
			err[BIN]=((ERR2)/((pes2.sum())**2))**0.5			
			Mcorr[BIN]=1+np.average(mcorr,weights=w2)
			if booterror_flag:
				error_et[BIN],error_ex[BIN],etboot0,exboot0 = qbootstrap_errors(shear,cero,w2,angle[maskr],nboot)
				etboot += [etboot0]
				exboot += [exboot0]
			else:
				error_et[BIN],error_ex[BIN] = err[BIN],err[BIN]
		
		
	return [R,SHEAR/Mcorr,CERO/Mcorr,err/Mcorr,nbin,error_et/Mcorr,error_ex/Mcorr,N]	

def sigma_C(redshifts):

	'''
	Computes sigma_c according to equation 10 Viola et al. 2015 (arXiv 1507.00735)
	for a array of sources at zs and lenses at zl
	
	Input:
		redshifts:	list of arrays [zl,zs,track,z_back]
		zl          (flt) array of the lenses redshifts
		zs          (flt) array of the sources redshifts
		track       (int) indexes array
		z_back      (flt) gap in z for the background galaxy selection
	Output:
	    list of [track,sigma_c]
	'''

	
	zl,zs,track,z_back = redshifts
			
	def integrando(z,zl,zs):	
	
		# Equation 10 Viola et al. 2015 (arXiv 1507.00735)
		
		sigma  = (1 + zs)*0.06	
		DL     = np.array(cosmo.angular_diameter_distance(zl))*1.e6*pc
		DLS    = np.array(cosmo.angular_diameter_distance_z1z2(zl,z))
		DS     = np.array(cosmo.angular_diameter_distance(z))
		beta   = DLS/DS
		
		# p_z equation ---
		
		p_z    = np.exp(-1.*((z-zs)**2/(2.*sigma**2)))/(sigma*np.sqrt(2.*np.pi))
		
		return DL*beta*p_z
		
	def expint(zl,zs):
		return quad(integrando, zl+z_back, np.inf, args=(zl,zs))[0]

	vec_expint = np.vectorize(expint)		 
	
	sigma_c    = (((cvel**2.0)/(4.0*np.pi*G))*(1./vec_expint(zl,zs)))*(pc**2/Msun)

	return np.array([track,sigma_c])
	
def Compute_sigma_c(cores,niter,outfile,catin,z_back):

	'''
	Splits the catalog to compute Sigma_c in parallel
	
	Input:
		cores:	    (int) Number of cores to parallelize
		niter:      (int) Number of slices in which the catalog is divided
		outfile:    (str) Name of the output file where [indexes,sigma_c] is recorded
		catin:      (str) Name of the input catalog
		z_back      (flt) gap in z for the background galaxy selection
	'''	
		
	hdulist = fits.open(catin)
	
	#hdulist = fits.open('gx_KDR2_clusters.fits')
	
	gxcat1  = hdulist[1].data	

	try:
		f       = open(outfile,'r')
		lines   = f.read().splitlines()
		last    = int(lines[-1].split(' ')[-2]) + 1
		f.close()
	except:
		last = 0
	        
	Z_c     = gxcat1.field('z')[last:]
	Z_B     = gxcat1.field('Z_B')[last:]
	
	del gxcat1
	del hdulist
	
	indices = np.arange(len(Z_c)) + last
	
	binini = 0
	step_iter = int(round(len(Z_c)/niter, 0))
	
	for n in range(niter):

		if n == (niter-1):
			zl    = Z_c[binini:]
			zs    = Z_B[binini:]
			track = indices[binini:]
		else: 
			zl    = Z_c[binini:binini+step_iter]
			zs    = Z_B[binini:binini+step_iter]
			track = indices[binini:binini+step_iter]
		
		print n,'/',niter,'Analizamos ', len(zl),'en paralelo'
		t1 = time.time()
	
		binini = binini + step_iter
	
		step = 0
		step0=int(round(len(zl)/cores, 0))
	
		entrada=[]
		
		for j in range(cores):
	
			if j==(cores-1):
				arreglo = [zl[step:], zs[step:], track[step:],z_back]
				entrada.append(arreglo)
			else: 
				arreglo = [zl[step:step+step0], zs[step:step+step0], track[step:step+step0],z_back]
				entrada.append(arreglo)
			
			step=step+step0
	
		pool = Pool(processes=(cores))
		salida = pool.map(sigma_C, entrada)
		pool.terminate()
		
		Sigma_C = salida[0]	
		
		for k in salida[1:]:
			Sigma_C = np.concatenate((Sigma_C,k),axis = 1)	
		
		f1=open(outfile,'a')
		np.savetxt(f1,Sigma_C.T,fmt = ['%12i']+['%12.6f'])
		f1.close()
		
		t2 = time.time()
		
		print 'Listo en...',(t2-t1)/60.

def profile(sample,RIN,ROUT,z_back,odds_min,snr_min,nmin,nmax,zmin,zmax,ndots):


	hdulist = fits.open('gx_KDR2_clusters.fits')
	
	gxcat1=hdulist[1].data
	
	del hdulist

	Z_c    = gxcat1.field('z')
	Z_B    = gxcat1.field('Z_B')
	ODDS   = gxcat1.field('ODDS')
	snr    = gxcat1.field('snr')
	N200   = gxcat1.field('N200')
	
	print '----- FILTRAR -----------'
	
	mask_back = (Z_B > (Z_c + z_back))*(ODDS >= odds_min)
	mask_lens = (snr > snr_min)*(N200 >= nmin)*(N200 <= nmax)*(Z_c >= zmin)*(Z_c <= zmax)
	mask = mask_back*mask_lens
	
	print '---- EXTRACT DATA -------'
	
	
	ra     = gxcat1.field('RAJ2000')[mask]
	dec    = gxcat1.field('DECJ2000')[mask]
		
	e1     = gxcat1.field('E1')[mask]
	e2     = gxcat1.field('E2')[mask]
		
	Z_c    = Z_c[mask]
		
	ID_c   = gxcat1.field('ID_c')[mask]
		
	peso   = gxcat1.field('WEIGHT')[mask]
	m      = gxcat1.field('M')[mask]
	
	ALFA0  = gxcat1.field('RA_bcg')[mask]
	DELTA0 = gxcat1.field('DEC_bcg')[mask]
	
	dls  = gxcat1.field('DLS')[mask]
	ds   = gxcat1.field('DS')[mask]
	dl   = gxcat1.field('DL')[mask]
	
	M200   = gxcat1.field('M200')[mask]
	eM200  = gxcat1.field('errM200')[mask]
	N200   = N200[mask]
	
	del(gxcat1)
	
	ides,index = np.unique(ID_c,return_index=True)
	
	
	NCUM = len(ides)
	
	print 'cantidad de CGs',NCUM
		
	
	KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	BETA_array = dls/ds
	beta       = BETA_array.mean()
	
	Dl = dl*1.e6*pc
	sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)
	
	
	print 'BETA.mean',beta
	
	SIGMAC = (((cvel**2.0)/(4.0*np.pi*G*Dl.mean())))*(pc**2/Msun)
	
	print 'SIGMA_C', SIGMAC
	
	
	rads, theta, test1,test2 = eq2p2(np.deg2rad(ra),
						np.deg2rad(dec),
						np.deg2rad(ALFA0),
						np.deg2rad(DELTA0))
	
	
	#Correct polar angle for e1, e2
	theta = theta+np.pi/2
	
	#get tangential ellipticities 
	et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
	#get cross ellipticities
	ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
	
	
	r=np.rad2deg(rads)*3600*KPCSCALE
	peso=peso/(sigma_c**2)
	
	
	STEP = (np.log10(ROUT)-np.log10(RIN))/ndots
	
	print 'STEP',STEP
	
	zmean=(Z_c).mean()
	zdisp=(Z_c).std()
	H=(70.0/(1.0e3*pc))*cd.e_z(zmean,**cosmo) #H at z_pair s-1
	roc=(3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
	roc_mpc=roc*((pc*1.0e6)**3.0)
	D_ang=cd.angular_diameter_distance(zmean, z0=0, **cosmo)
	kpcscale=D_ang*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	print '---------------------------------------------------------'
	print '             COMPUTING THE SHEAR PROFILES                '
	print '========================================================='
	
	profile = shear_profile_log(RIN,ROUT,r,et,ex,peso,m,STEP,sigma_c,'si')
		
		

	R=profile[0]/1.0e3 #r en Mpc
	
	
	shear  = profile[1]
	cero   = profile[2]
	BIN    = profile[4]
	err_et = profile[5]
	err_ex = profile[6]
	
	print '---------------------------------------------------------'
	print '                   FITTING PROFILES                      '
	print '========================================================='
	
	
	print 'First a SIS profile'

	sis=SIS_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN])

	DISP      = sis[0]
	ERRORDISP = sis[1]
	CHI_sis   = sis[2]
	X         = sis[3]
	Y         = sis[4]

	M200_SIS=((2.*(DISP*1.e3)**3)/((50**0.5)*G*H))/(Msun)
	e_m200_SIS=(((6.*(DISP*1.e3)**2)/((50**0.5)*G*H))*(ERRORDISP*1.e3))/(Msun)
	
	print 'Sigma =', '%.2e' % DISP, '+/-','%.2e' % ERRORDISP
	print 'M_200_SIS =', '%.2e' % M200_SIS, '+/-','%.2e' % e_m200_SIS

	print 'Now is trying to fit a NFW profile...'

	nfw        = NFW_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN],zmean,roc)
	c          = nfw[5]
	CHI_nfw    = nfw[2]
	RS         = nfw[0]/c
	R200       = nfw[0]
	error_R200 = nfw[1]
	x2         = nfw[3]
	y2         = nfw[4]

	M200_NFW   = (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)
	e_M200_NFW = ((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*error_R200
	
	print 'R200 =',R200,'+/-', '%.2f' % error_R200,'Mpc'
	print 'M200 nfw =',M200_NFW,'+/-',e_M200_NFW


	#------------------------------------------------------
	# ------- FOR THE PLOT ------

	x=np.zeros(2,float)
	y=np.zeros(2,float)
	x[1]=10000.

	rcParams['font.family'] = 'serif'
	#rcParams['figure.figsize'] =11.,5.
	pylab.rcParams.update()
	majorFormatter = FormatStrFormatter('%.1f')
	fig = plt.figure(figsize=(8, 6))  #tamano del plot
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) #divide en 2 el eje x, en 1 el eje y y da la razon de alturas

	#asigna los sublots

	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	#grafica


	blancox=5000.
	blancoy=5000.

	name_label = sample


	ax.plot(R,shear,'ko')
	ax.plot(blancox,blancoy,'w.',label=name_label)#

	ax.legend(loc=1,frameon=False)
	ax.plot(x2,y2,'r--',label='NFW: c='+str('%.1f' % c)+', $R_{200}$ = '+str('%.2f' % R200)+' $\pm$ '+str('%.2f' % error_R200)+' Mpc$\,h^{-1}_{70}$, $\chi_{red}^{2} =$'+str('%.1f' % CHI_nfw)) #, c='+str(round(c)))
	#ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,-1)))+' $\pm$ '+str(int(round(ERRORDISP,-1)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,0)))+' $\pm$ '+str(int(round(ERRORDISP,0)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.errorbar(R, shear, yerr=err_et, fmt=None, ecolor='k')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax.legend(loc=1,frameon=False)


	# axis detail
	ax.axis([RIN/1000.,ROUT/1000.,1.,5000.])
	ax.set_xscale('log', nonposy='clip')
	ax.set_yscale('log', nonposy='clip')
	ax.xaxis.set_ticks(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.set_xticklabels(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.yaxis.set_ticks(np.arange(10., 200., 100.))
	ax.set_yticklabels(np.arange(10., 200., 100.))

	#label					
	ax.set_ylabel(u'$\Delta\Sigma_{\parallel} (M_{\odot}\,pc^{-2})$',fontsize=15)

	#-----------------------------

	ax2.plot(x,y,'k')
	ax2.plot(R,cero,'kx')
	ax2.errorbar(R,cero, yerr=err_ex, fmt=None, ecolor='k')

	#axis details
	ax2.axis([RIN/1000.,ROUT/1000.,-25.,30.])
	ax2.yaxis.set_ticks(np.arange(-20., 20.,15.))
	ax2.set_yticklabels(np.arange(-20., 20., 15.))
	ax2.set_xscale('log', nonposx='clip')
	#ax2.xaxis.set_ticks([0.2,0.5,1,2,3,4,5])
	#ax2.set_xticklabels([0.2,0.5,1,2,3,4,5])
	#ax2.xaxis.set_ticks([0.5,1.0,1.5,2.0,2.5])
	#ax2.set_xticklabels([0.5,1.0,1.5,2.0,2.5])
	ax2.xaxis.set_ticks([0.1,0.2,0.5,1.0,5.0])
	ax2.set_xticklabels([0.1,0.2,0.5,1.0,5.0])



	#labels
	ax2.set_ylabel(r'$\Delta\Sigma_{\times} $',fontsize=15)
	ax2.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)

	#to join the plots
	fig.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)


	plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
	plotname = 'shear_profile_'+sample+'.eps'
	plt.savefig(plotname, format='eps',bbox_inches='tight')
	plt.show()		
	

		
	f1=open('profile'+sample+'_rcut.cat','w')
	f1.write('# z_mean = '+str('%.2f' % zmean)+' \n')
	f1.write('# z_back = '+str('%.2f' % z_back)+' \n')
	f1.write('# odds_min = '+str('%.1f' % odds_min)+' \n')
	f1.write('# snr_min = '+str('%.1f' % snr_min)+' \n')
	f1.write('# N200_min = '+str(np.int(nmin))+' \n')
	f1.write('# N200_max = '+str(np.int(nmax))+' \n')
	f1.write('# z_min = '+str('%.1f' % zmin)+' \n')
	f1.write('# z_max = '+str('%.1f' % zmax)+' \n')
	f1.write('# R,shear,err_et,cero,err_ex \n')
	profile = np.column_stack((R[:BIN],shear[:BIN],err_et[:BIN],cero[:BIN],err_ex[:BIN]))
	np.savetxt(f1,profile,fmt = ['%12.6f']*5)
	f1.close()
	
	f1=open('out.cat','a')
	f1.write(str('%.2f' % zmean)+'  ')
	f1.write(str('%.2f' % z_back)+'  ')
	f1.write(str('%.1f' % odds_min)+'  ')
	f1.write(str('%.1f' % snr_min)+'  ')
	f1.write(str(np.int(nmin))+'  ')
	f1.write(str(np.int(nmax))+'  ')
	f1.write(str('%.1f' % zmin)+'  ')
	f1.write(str('%.1f' % zmax)+'  ')
	f1.write(name_label+' '+str(int(NCUM))+' ') 
	f1.write(str('%.1f' % (N200.mean()))+' ')
	f1.write(str('%.1f' % (M200.mean()))+' ')
	f1.write(str('%.1f' % (eM200.mean()))+' ')
	f1.write(str(int(round(DISP,0)))+'  '+str(int(round(ERRORDISP,0)))+' ')
	f1.write(str('%.1f' % (M200_SIS/1.e12))+' '+str('%.1f' % (e_m200_SIS/1.e12))+' ')
	f1.write(str('%.1f' % CHI_sis)+' ')
	f1.write(str('%.2f' % R200)+' '+str('%.2f' % error_R200)+' ')
	f1.write(str('%.1f' % (M200_NFW/1.e12))+' '+str('%.1f' % (e_M200_NFW/1.e12))+' ')
	f1.write(str('%.1f' % CHI_nfw)+' \n')
	f1.close()


def profile_redMapper_KiDS(sample,RIN,ROUT,ndots,z_back,odds_min,lmin,lmax,zmin,zmax,Sc):

	'''
	Make the profile for redMapper clusters with background galaxies from KiDS.
	
	Input:
		sample:		(str) Name of the sample
		----- profile parameters ----
		RIN: 	    (flt) Inner radius in kpc
		ROUT: 	    (flt) Outer radius in kpc
		ndots:      (int) Number of bins in the profile
		----- background galaxies cut --- 
		z_back: 	(flt) Gap for the background galaxy selection
		odds_min: 	(flt) Lower ODDS cut
		----- lens cut ----
		lmin: 		(flt) Min lambda for the selected groups
		lmax: 		(flt) Max lambda for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		---- Sigma_c ----
		Sc:         (bool) True: Uses the sigma_c from the catalog, 
		                   False: Computes it with the redshifts
	'''

	hdulist = fits.open('gx_redMapper2.fits')
	
	gxcat1=hdulist[1].data
	
	del hdulist

	Z_c    = gxcat1.field('z')
	Z_B    = gxcat1.field('Z_B')
	ODDS   = gxcat1.field('ODDS')
	lamb   = gxcat1.field('lambda')
	
	
	print '---- FILTRAR -------'
	
	mask_back = (Z_B > (Z_c + z_back))*(ODDS >= odds_min)*(Z_B < 0.9)
	mask_lens = (lamb >= lmin)*(lamb < lmax)*(Z_c >= zmin)*(Z_c < zmax)
	mask = mask_back*mask_lens
	
	print '---- EXTRACT DATA -------'
	
	
	ra     = gxcat1.field('RAJ2000')[mask]
	dec    = gxcat1.field('DECJ2000')[mask]
		
	e1     = gxcat1.field('E1')[mask]
	e2     = gxcat1.field('E2')[mask]
		
	Z_c    = Z_c[mask]
		
	ID_c   = gxcat1.field('ID_c')[mask]
		
	peso   = gxcat1.field('WEIGHT')[mask]
	m      = gxcat1.field('M')[mask]
	
	ALFA0  = gxcat1.field('RA')[mask]
	DELTA0 = gxcat1.field('DEC')[mask]
	
	dls  = gxcat1.field('DLS')[mask]
	ds   = gxcat1.field('DS')[mask]
	dl   = gxcat1.field('DL')[mask]
	lamb = lamb[mask]
	

	
	ides,index = np.unique(ID_c,return_index=True)
	
	
	NCUM = len(ides)
	
	print 'cantidad de lentes',NCUM
		
	
	KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	BETA_array = dls/ds
	beta       = BETA_array.mean()
	
	Dl = dl*1.e6*pc
	
	
	
	print 'BETA.mean',beta
	
	# COMPUTE Sigma_C
	
	if Sc:
		sigma_c = (gxcat1.field('Sigma_c')[mask]/(1.e6*pc)) # read from catalog
	else:
		sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun) # computes it according to z
		
	SIGMAC = (((cvel**2.0)/(4.0*np.pi*G*Dl.mean())))*(pc**2/Msun)
	print 'SIGMA_C', SIGMAC

	del(gxcat1)	
	
	
	# COMPUTE ANGLES
	
	rads, theta, test1,test2 = eq2p2(np.deg2rad(ra),
						np.deg2rad(dec),
						np.deg2rad(ALFA0),
						np.deg2rad(DELTA0))
	
	
	#Correct polar angle for e1, e2
	theta = theta+np.pi/2
	
	#get tangential ellipticities 
	et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
	#get cross ellipticities
	ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
	
	
	r=np.rad2deg(rads)*3600*KPCSCALE # distance to the lens in kpc
	peso=peso/(sigma_c**2)
	
	# COMPUTES THE STEP
	
	STEP = (np.log10(ROUT)-np.log10(RIN))/ndots
	
	print 'STEP',STEP
	
	# COSMOLOGICAL PARAMETERS
	
	
	zmean    = (Z_c).mean()
	zdisp    = (Z_c).std()
	H        = cosmo.H(zmean).value/(1.0e3*pc) #H at z_pair s-1 
	roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
	roc_mpc  = roc*((pc*1.0e6)**3.0)
	D_ang    = cosmo.angular_diameter_distance(zmean)
	kpcscale = D_ang*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	
	print '---------------------------------------------------------'
	print '             COMPUTING THE SHEAR PROFILES                '
	print '========================================================='
	
	profile = shear_profile_log(RIN,ROUT,r,et,ex,peso,m,STEP,sigma_c,'si')
	R=profile[0]/1.0e3 #r en Mpc
	
	
	shear  = profile[1]
	cero   = profile[2]
	BIN    = profile[4]
	err_et = profile[5]
	err_ex = profile[6]
	
	print '---------------------------------------------------------'
	print '                   FITTING PROFILES                      '
	print '========================================================='
	
	
	print 'First a SIS profile'

	sis=SIS_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN])

	DISP      = sis[0]
	ERRORDISP = sis[1]
	CHI_sis   = sis[2]
	X         = sis[3]
	Y         = sis[4]

	M200_SIS=((2.*(DISP*1.e3)**3)/((50**0.5)*G*H))/(Msun)
	e_m200_SIS=(((6.*(DISP*1.e3)**2)/((50**0.5)*G*H))*(ERRORDISP*1.e3))/(Msun)
	
	print 'Sigma =', '%.2e' % DISP, '+/-','%.2e' % ERRORDISP
	print 'M_200_SIS =', '%.2e' % M200_SIS, '+/-','%.2e' % e_m200_SIS

	print 'Now is trying to fit a NFW profile...'

	nfw        = NFW_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN],zmean,roc)
	c          = nfw[5]
	CHI_nfw    = nfw[2]
	RS         = nfw[0]/c
	R200       = nfw[0]
	error_R200 = nfw[1]
	x2         = nfw[3]
	y2         = nfw[4]

	M200_NFW   = (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)
	e_M200_NFW = ((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*error_R200
	
	print 'R200 =',R200,'+/-', '%.2f' % error_R200,'Mpc'
	print 'M200 nfw =',M200_NFW,'+/-',e_M200_NFW


	#------------------------------------------------------
	# ------- FOR THE PLOT ------

	x=np.zeros(2,float)
	y=np.zeros(2,float)
	x[1]=10000.

	rcParams['font.family'] = 'serif'
	#rcParams['figure.figsize'] =11.,5.
	pylab.rcParams.update()
	majorFormatter = FormatStrFormatter('%.1f')
	fig = plt.figure(figsize=(8, 6))  #tamano del plot
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) #divide en 2 el eje x, en 1 el eje y y da la razon de alturas

	#asigna los sublots

	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	#grafica


	blancox=5000.
	blancoy=5000.

	name_label = sample


	ax.plot(R,shear,'ko')
	ax.plot(blancox,blancoy,'w.',label=name_label)#

	ax.legend(loc=1,frameon=False, scatterpoints = 1)
	ax.plot(x2,y2,'r--',label='NFW: c='+str('%.1f' % c)+', $R_{200}$ = '+str('%.2f' % R200)+' $\pm$ '+str('%.2f' % error_R200)+' Mpc$\,h^{-1}_{70}$, $\chi_{red}^{2} =$'+str('%.1f' % CHI_nfw)) #, c='+str(round(c)))
	#ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,-1)))+' $\pm$ '+str(int(round(ERRORDISP,-1)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,0)))+' $\pm$ '+str(int(round(ERRORDISP,0)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.errorbar(R, shear, yerr=err_et, fmt=None, ecolor='k')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax.legend(loc=1,frameon=False, scatterpoints = 1)


	# axis detail
	ax.axis([RIN/1000.,ROUT/1000.,1.,5000.])
	ax.set_xscale('log', nonposy='clip')
	ax.set_yscale('log', nonposy='clip')
	ax.xaxis.set_ticks(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.set_xticklabels(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.yaxis.set_ticks(np.arange(10., 200., 100.))
	ax.set_yticklabels(np.arange(10., 200., 100.))

	#label					
	ax.set_ylabel(u'$\Delta\Sigma_{\parallel} (M_{\odot}\,pc^{-2})$',fontsize=15)

	#-----------------------------

	ax2.plot(x,y,'k')
	ax2.plot(R,cero,'kx')
	ax2.errorbar(R,cero, yerr=err_ex, fmt=None, ecolor='k')

	#axis details
	ax2.axis([RIN/1000.,ROUT/1000.,-25.,30.])
	ax2.yaxis.set_ticks(np.arange(-20., 20.,15.))
	ax2.set_yticklabels(np.arange(-20., 20., 15.))
	ax2.set_xscale('log', nonposx='clip')
	#ax2.xaxis.set_ticks([0.2,0.5,1,2,3,4,5])
	#ax2.set_xticklabels([0.2,0.5,1,2,3,4,5])
	#ax2.xaxis.set_ticks([0.5,1.0,1.5,2.0,2.5])
	#ax2.set_xticklabels([0.5,1.0,1.5,2.0,2.5])
	ax2.xaxis.set_ticks([0.1,0.2,0.5,1.0,5.0])
	ax2.set_xticklabels([0.1,0.2,0.5,1.0,5.0])



	#labels
	ax2.set_ylabel(r'$\Delta\Sigma_{\times} $',fontsize=15)
	ax2.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)

	#to join the plots
	fig.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)


	plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
	plotname = 'shear_profile_'+sample+'_KiDS.eps'
	plt.savefig(plotname, format='eps',bbox_inches='tight')
	plt.show()		
	

		
	f1=open('profile'+sample+'_KiDS.cat','w')
	f1.write('# z_mean = '+str('%.2f' % zmean)+' \n')
	f1.write('# z_back = '+str('%.2f' % z_back)+' \n')
	f1.write('# odds_min = '+str('%.1f' % odds_min)+' \n')
	f1.write('# lambda_min = '+str('%.1f' %  lmin)+' \n')
	f1.write('# lambda_max = '+str('%.1f' %  lmax)+' \n')
	f1.write('# z_min = '+str('%.1f' % zmin)+' \n')
	f1.write('# z_max = '+str('%.1f' % zmax)+' \n')
	f1.write('# R,shear,err_et,cero,err_ex \n')
	profile = np.column_stack((R[:BIN],shear[:BIN],err_et[:BIN],cero[:BIN],err_ex[:BIN]))
	np.savetxt(f1,profile,fmt = ['%12.6f']*5)
	f1.close()
	
	f1=open('out_KiDS.cat','a')
	f1.write(str('%.2f' % zmean)+'  ')
	f1.write(str('%.2f' % z_back)+'  ')
	f1.write(str('%.1f' % odds_min)+'  ')
	f1.write(str('%.1f' %  lmin)+'  ')
	f1.write(str('%.1f' %  lmax)+'  ')
	f1.write(str('%.1f' % zmin)+'  ')
	f1.write(str('%.1f' % zmax)+'  ')
	f1.write(name_label+' '+str(int(NCUM))+' ') 
	f1.write(str('%.1f' % (lamb.mean()))+' ')
	f1.write(str(int(round(DISP,0)))+'  '+str(int(round(ERRORDISP,0)))+' ')
	f1.write(str('%.1f' % (M200_SIS/1.e12))+' '+str('%.1f' % (e_m200_SIS/1.e12))+' ')
	f1.write(str('%.1f' % CHI_sis)+' ')
	f1.write(str('%.2f' % R200)+' '+str('%.2f' % error_R200)+' ')
	f1.write(str('%.1f' % (M200_NFW/1.e12))+' '+str('%.1f' % (e_M200_NFW/1.e12))+' ')
	f1.write(str('%.1f' % CHI_nfw)+' \n')
	f1.close()	

	
def profile_redMapper_CS82(sample,RIN,ROUT,ndots,z_back,odds_min,lmin,lmax,zmin,zmax,Sc):

	'''
	Make the profile for redMapper clusters with background galaxies from CS82.
	
	Input:
		sample:		(str) Name of the sample
		----- profile parameters ----
		RIN: 	    (flt) Inner radius in kpc
		ROUT: 	    (flt) Outer radius in kpc
		ndots:      (int) Number of bins in the profile
		----- background galaxies cut --- 
		z_back: 	(flt) Gap for the background galaxy selection
		odds_min: 	(flt) Lower ODDS cut
		----- lens cut ----
		lmin: 		(flt) Min lambda for the selected groups
		lmax: 		(flt) Max lambda for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		---- Sigma_c ----
		Sc:         (bool) True: Uses the sigma_c from the catalog, 
		                   False: Computes it with the redshifts
	'''


	hdulist = fits.open('gx_redMapper_CS82.fits')
	
	gxcat1=hdulist[1].data
	
	del hdulist

	Z_c    = gxcat1.field('z')
	Z_B    = gxcat1.field('Z_B')
	ODDS   = gxcat1.field('ODDS')
	lamb   = gxcat1.field('lambda')
	ALFA0  = gxcat1.field('RA')
	DELTA0 = gxcat1.field('DEC')
	#s95    = gxcat1.field('BPZ_LOW95')
	
	print '---- FILTRAR -------'
	
	mask_back = (Z_B > (Z_c + z_back))*(ODDS >= odds_min)#*(Z_B > (Z_c + s95/2.))
	mgroup = (DELTA0>-1.25)*(DELTA0<1.25)*(ALFA0<45.)*(ALFA0>-42.)
	mask_lens = (lamb >= lmin)*(lamb < lmax)*(Z_c >= zmin)*(Z_c < zmax)
	mask = mask_back*mask_lens*mgroup
	
	print '---- EXTRACT DATA -------'
	
	
	ra     = gxcat1.field('RAJ2000')[mask]
	dec    = gxcat1.field('DECJ2000')[mask]
		
	e1     = gxcat1.field('E1')[mask]
	e2     = gxcat1.field('E2')[mask]
		
	Z_c    = Z_c[mask]
		
	ID_c   = gxcat1.field('ID_c')[mask]
		
	peso   = gxcat1.field('WEIGHT')[mask]
	m      = gxcat1.field('M')[mask]
	
	ALFA0  = ALFA0[mask]
	DELTA0 = DELTA0[mask]
	
	dls  = gxcat1.field('DLS')[mask]
	ds   = gxcat1.field('DS')[mask]
	dl   = gxcat1.field('DL')[mask]
	lamb = lamb[mask]
	

	
	ides,index = np.unique(ID_c,return_index=True)
	
	
	NCUM = len(ides)
	
	print 'cantidad de lentes',NCUM
		
	
	KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	BETA_array = dls/ds
	beta       = BETA_array.mean()
	
	Dl = dl*1.e6*pc
	
	
	
	print 'BETA.mean',beta
	
	# COMPUTE Sigma_C
	
	if Sc:
		sigma_c = (gxcat1.field('Sigma_c')[mask]/(1.e6*pc)) # read from catalog
	else:
		sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun) # computes it according to z
		
	SIGMAC = (((cvel**2.0)/(4.0*np.pi*G*Dl.mean())))*(pc**2/Msun)
	print 'SIGMA_C', SIGMAC

	del(gxcat1)	
	
	
	# COMPUTE ANGLES
	
	rads, theta, test1,test2 = eq2p2(np.deg2rad(ra),
						np.deg2rad(dec),
						np.deg2rad(ALFA0),
						np.deg2rad(DELTA0))
	
	
	#Correct polar angle for e1, e2
	theta = theta+np.pi/2
	
	#get tangential ellipticities 
	et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
	#get cross ellipticities
	ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
	
	
	r=np.rad2deg(rads)*3600*KPCSCALE # distance to the lens in kpc
	peso=peso/(sigma_c**2)
	
	# COMPUTES THE STEP
	
	STEP = (np.log10(ROUT)-np.log10(RIN))/ndots
	
	print 'STEP',STEP
	
	# COSMOLOGICAL PARAMETERS
	
	
	zmean    = (Z_c).mean()
	zdisp    = (Z_c).std()
	H        = cosmo.H(zmean).value/(1.0e3*pc) #H at z_pair s-1 
	roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
	roc_mpc  = roc*((pc*1.0e6)**3.0)
	D_ang    = cosmo.angular_diameter_distance(zmean)
	kpcscale = D_ang*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	
	print '---------------------------------------------------------'
	print '             COMPUTING THE SHEAR PROFILES                '
	print '========================================================='
	
	profile = shear_profile_log(RIN,ROUT,r,et,ex,peso,m,STEP,sigma_c,'si')
	R=profile[0]/1.0e3 #r en Mpc
	
	
	shear  = profile[1]
	cero   = profile[2]
	BIN    = profile[4]
	err_et = profile[5]
	err_ex = profile[6]
	
	print '---------------------------------------------------------'
	print '                   FITTING PROFILES                      '
	print '========================================================='
	
	
	print 'First a SIS profile'

	sis=SIS_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN])

	DISP      = sis[0]
	ERRORDISP = sis[1]
	CHI_sis   = sis[2]
	X         = sis[3]
	Y         = sis[4]

	M200_SIS=((2.*(DISP*1.e3)**3)/((50**0.5)*G*H))/(Msun)
	e_m200_SIS=(((6.*(DISP*1.e3)**2)/((50**0.5)*G*H))*(ERRORDISP*1.e3))/(Msun)
	
	print 'Sigma =', '%.2e' % DISP, '+/-','%.2e' % ERRORDISP
	print 'M_200_SIS =', '%.2e' % M200_SIS, '+/-','%.2e' % e_m200_SIS

	print 'Now is trying to fit a NFW profile...'

	nfw        = NFW_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN],zmean,roc)
	c          = nfw[5]
	CHI_nfw    = nfw[2]
	RS         = nfw[0]/c
	R200       = nfw[0]
	error_R200 = nfw[1]
	x2         = nfw[3]
	y2         = nfw[4]

	M200_NFW   = (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)
	e_M200_NFW = ((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*error_R200
	
	print 'R200 =',R200,'+/-', '%.2f' % error_R200,'Mpc'
	print 'M200 nfw =',M200_NFW,'+/-',e_M200_NFW


	#------------------------------------------------------
	# ------- FOR THE PLOT ------

	x=np.zeros(2,float)
	y=np.zeros(2,float)
	x[1]=10000.

	rcParams['font.family'] = 'serif'
	#rcParams['figure.figsize'] =11.,5.
	pylab.rcParams.update()
	majorFormatter = FormatStrFormatter('%.1f')
	fig = plt.figure(figsize=(8, 6))  #tamano del plot
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) #divide en 2 el eje x, en 1 el eje y y da la razon de alturas

	#asigna los sublots

	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	#grafica


	blancox=5000.
	blancoy=5000.

	name_label = sample


	ax.plot(R,shear,'ko')
	ax.plot(blancox,blancoy,'w.',label=name_label)#

	ax.legend(loc=1,frameon=False, scatterpoints = 1)
	ax.plot(x2,y2,'r--',label='NFW: c='+str('%.1f' % c)+', $R_{200}$ = '+str('%.2f' % R200)+' $\pm$ '+str('%.2f' % error_R200)+' Mpc$\,h^{-1}_{70}$, $\chi_{red}^{2} =$'+str('%.1f' % CHI_nfw)) #, c='+str(round(c)))
	#ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,-1)))+' $\pm$ '+str(int(round(ERRORDISP,-1)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,0)))+' $\pm$ '+str(int(round(ERRORDISP,0)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.errorbar(R, shear, yerr=err_et, fmt=None, ecolor='k')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax.legend(loc=1,frameon=False, scatterpoints = 1)


	# axis detail
	ax.axis([RIN/1000.,ROUT/1000.,1.,5000.])
	ax.set_xscale('log', nonposy='clip')
	ax.set_yscale('log', nonposy='clip')
	ax.xaxis.set_ticks(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.set_xticklabels(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.yaxis.set_ticks(np.arange(10., 200., 100.))
	ax.set_yticklabels(np.arange(10., 200., 100.))

	#label					
	ax.set_ylabel(u'$\Delta\Sigma_{\parallel} (M_{\odot}\,pc^{-2})$',fontsize=15)

	#-----------------------------

	ax2.plot(x,y,'k')
	ax2.plot(R,cero,'kx')
	ax2.errorbar(R,cero, yerr=err_ex, fmt=None, ecolor='k')

	#axis details
	ax2.axis([RIN/1000.,ROUT/1000.,-25.,30.])
	ax2.yaxis.set_ticks(np.arange(-20., 20.,15.))
	ax2.set_yticklabels(np.arange(-20., 20., 15.))
	ax2.set_xscale('log', nonposx='clip')
	#ax2.xaxis.set_ticks([0.2,0.5,1,2,3,4,5])
	#ax2.set_xticklabels([0.2,0.5,1,2,3,4,5])
	#ax2.xaxis.set_ticks([0.5,1.0,1.5,2.0,2.5])
	#ax2.set_xticklabels([0.5,1.0,1.5,2.0,2.5])
	ax2.xaxis.set_ticks([0.1,0.2,0.5,1.0,5.0])
	ax2.set_xticklabels([0.1,0.2,0.5,1.0,5.0])



	#labels
	ax2.set_ylabel(r'$\Delta\Sigma_{\times} $',fontsize=15)
	ax2.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)

	#to join the plots
	fig.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)


	plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
	plotname = 'shear_profile_'+sample+'_CS82.eps'
	plt.savefig(plotname, format='eps',bbox_inches='tight')
	plt.show()		
	

		
	f1=open('profile'+sample+'_CS82.cat','w')
	f1.write('# z_mean = '+str('%.2f' % zmean)+' \n')
	f1.write('# z_back = '+str('%.2f' % z_back)+' \n')
	f1.write('# odds_min = '+str('%.1f' % odds_min)+' \n')
	f1.write('# lambda_min = '+str('%.1f' %  lmin)+' \n')
	f1.write('# lambda_max = '+str('%.1f' %  lmax)+' \n')
	f1.write('# z_min = '+str('%.1f' % zmin)+' \n')
	f1.write('# z_max = '+str('%.1f' % zmax)+' \n')
	f1.write('# R,shear,err_et,cero,err_ex \n')
	profile = np.column_stack((R[:BIN],shear[:BIN],err_et[:BIN],cero[:BIN],err_ex[:BIN]))
	np.savetxt(f1,profile,fmt = ['%12.6f']*5)
	f1.close()
	
	f1=open('out_CS82.cat','a')
	f1.write(str('%.2f' % zmean)+'  ')
	f1.write(str('%.2f' % z_back)+'  ')
	f1.write(str('%.1f' % odds_min)+'  ')
	f1.write(str('%.1f' %  lmin)+'  ')
	f1.write(str('%.1f' %  lmax)+'  ')
	f1.write(str('%.1f' % zmin)+'  ')
	f1.write(str('%.1f' % zmax)+'  ')
	f1.write(name_label+' '+str(int(NCUM))+' ') 
	f1.write(str('%.1f' % (lamb.mean()))+' ')
	f1.write(str(int(round(DISP,0)))+'  '+str(int(round(ERRORDISP,0)))+' ')
	f1.write(str('%.1f' % (M200_SIS/1.e12))+' '+str('%.1f' % (e_m200_SIS/1.e12))+' ')
	f1.write(str('%.1f' % CHI_sis)+' ')
	f1.write(str('%.2f' % R200)+' '+str('%.2f' % error_R200)+' ')
	f1.write(str('%.1f' % (M200_NFW/1.e12))+' '+str('%.1f' % (e_M200_NFW/1.e12))+' ')
	f1.write(str('%.1f' % CHI_nfw)+' \n')
	f1.close()	
	
	
def profile_redMapper_combined(sample,RIN,ROUT,ndots,z_back,odds_min,lmin,lmax,zmin,zmax,Sc):
	
	'''
	Make the profile for redMapper clusters. Combine CS82 and KiDS catalogs.
	
	Input:
		sample:		(str) Name of the sample
		----- profile parameters ----
		RIN: 	    (flt) Inner radius in kpc
		ROUT: 	    (flt) Outer radius in kpc
		ndots:      (int) Number of bins in the profile
		----- background galaxies cut --- 
		z_back: 	(flt) Gap for the background galaxy selection
		odds_min: 	(flt) Lower ODDS cut
		----- lens cut ----
		lmin: 		(flt) Min lambda for the selected groups
		lmax: 		(flt) Max lambda for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		zmin:       (flt) Min redshift for the selected groups
		---- Sigma_c ----
		Sc:         (bool) True: Uses the sigma_c from the catalog, 
		                   False: Computes it with the redshifts
	'''

	# READ THE CATALOGS
	
	hdulist = fits.open('gx_redMapper_CS82.fits')
	
	gxcat1=hdulist[1].data
	
	del hdulist
	
	hdulist = fits.open('gx_redMapper.fits')
	
	gxcat2=hdulist[1].data
	
	del hdulist
	
	# EXTRACT SOME PARAMETERS

	Z_c    = np.concatenate((gxcat1.field('z'),gxcat2.field('z')))
	Z_B    = np.concatenate((gxcat1.field('Z_B'),gxcat2.field('Z_B')))
	ODDS   = np.concatenate((gxcat1.field('ODDS'),gxcat2.field('ODDS')))
	lamb   = np.concatenate((gxcat1.field('lambda'),gxcat2.field('lambda'))) 
	
	
	print '---- FILTRAR -------'
	
	mask_back = (Z_B > (Z_c + z_back))*(ODDS >= odds_min)*(Z_B < 0.9)
	mask_lens = (lamb >= lmin)*(lamb < lmax)*(Z_c >= zmin)*(Z_c < zmax)
	mask = mask_back*mask_lens
	
	print '---- EXTRACT DATA -------'
	
	
	ra     = np.concatenate((gxcat1.field('RAJ2000'),gxcat2.field('RAJ2000')))[mask]
	dec    = np.concatenate((gxcat1.field('DECJ2000'),gxcat2.field('DECJ2000')))[mask]
		
	e1     = np.concatenate((gxcat1.field('E1'),gxcat2.field('E1')))[mask]
	e2     = np.concatenate((gxcat1.field('E2'),gxcat2.field('E2')))[mask]
		
	Z_c    = Z_c[mask]
		
	ID_c   = np.concatenate((gxcat1.field('ID_c'),gxcat2.field('ID_c')))[mask]
		
	peso   = np.concatenate((gxcat1.field('WEIGHT'),gxcat2.field('WEIGHT')))[mask]
	m      = np.concatenate((gxcat1.field('M'),gxcat2.field('M')))[mask]
	
	ALFA0  = np.concatenate((gxcat1.field('RA'),gxcat2.field('RA')))[mask]
	DELTA0 = np.concatenate((gxcat1.field('DEC'),gxcat2.field('DEC')))[mask]
	
	dls  = np.concatenate((gxcat1.field('DLS'),gxcat2.field('DLS')))[mask]
	ds   = np.concatenate((gxcat1.field('DS'),gxcat2.field('DS')))[mask]
	dl   = np.concatenate((gxcat1.field('DL'),gxcat2.field('DL')))[mask]
	lamb = lamb[mask]
	

	
	ides,index = np.unique(ID_c,return_index=True)
	
	
	NCUM = len(ides)
	
	print 'cantidad de lentes',NCUM
		
	
	KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	BETA_array = dls/ds
	beta       = BETA_array.mean()
	
	Dl = dl*1.e6*pc
	
	
	
	print 'BETA.mean',beta
	
	# COMPUTE Sigma_C
	
	if Sc:
		sigma_c = (gxcat1.field('Sigma_c')[mask]/(1.e6*pc)) # read from catalog
	else:
		sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun) # computes it according to z
		
	SIGMAC = (((cvel**2.0)/(4.0*np.pi*G*Dl.mean())))*(pc**2/Msun)
	print 'SIGMA_C', SIGMAC

	del(gxcat1)	
	
	
	# COMPUTE ANGLES
	
	rads, theta, test1,test2 = eq2p2(np.deg2rad(ra),
						np.deg2rad(dec),
						np.deg2rad(ALFA0),
						np.deg2rad(DELTA0))
	
	
	#Correct polar angle for e1, e2
	theta = theta+np.pi/2
	
	#get tangential ellipticities 
	et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
	#get cross ellipticities
	ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
	
	
	r=np.rad2deg(rads)*3600*KPCSCALE # distance to the lens in kpc
	peso=peso/(sigma_c**2)
	
	# COMPUTES THE STEP
	
	STEP = (np.log10(ROUT)-np.log10(RIN))/ndots
	
	print 'STEP',STEP
	
	# COSMOLOGICAL PARAMETERS
	
	
	zmean    = (Z_c).mean()
	zdisp    = (Z_c).std()
	H        = cosmo.H(zmean).value/(1.0e3*pc) #H at z_pair s-1 
	roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
	roc_mpc  = roc*((pc*1.0e6)**3.0)
	D_ang    = cosmo.angular_diameter_distance(zmean)
	kpcscale = D_ang*(((1.0/3600.0)*np.pi)/180.0)*1000.0
	
	print '---------------------------------------------------------'
	print '             COMPUTING THE SHEAR PROFILES                '
	print '========================================================='
	
	profile = shear_profile_log(RIN,ROUT,r,et,ex,peso,m,STEP,sigma_c,'si')
	R=profile[0]/1.0e3 #r en Mpc
	
	
	shear  = profile[1]
	cero   = profile[2]
	BIN    = profile[4]
	err_et = profile[5]
	err_ex = profile[6]
	
	print '---------------------------------------------------------'
	print '                   FITTING PROFILES                      '
	print '========================================================='
	
	
	print 'First a SIS profile'

	sis=SIS_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN])

	DISP      = sis[0]
	ERRORDISP = sis[1]
	CHI_sis   = sis[2]
	X         = sis[3]
	Y         = sis[4]

	M200_SIS=((2.*(DISP*1.e3)**3)/((50**0.5)*G*H))/(Msun)
	e_m200_SIS=(((6.*(DISP*1.e3)**2)/((50**0.5)*G*H))*(ERRORDISP*1.e3))/(Msun)
	
	print 'Sigma =', '%.2e' % DISP, '+/-','%.2e' % ERRORDISP
	print 'M_200_SIS =', '%.2e' % M200_SIS, '+/-','%.2e' % e_m200_SIS

	print 'Now is trying to fit a NFW profile...'

	nfw        = NFW_stack_fit(R[:BIN],shear[:BIN],err_et[:BIN],zmean,roc)
	c          = nfw[5]
	CHI_nfw    = nfw[2]
	RS         = nfw[0]/c
	R200       = nfw[0]
	error_R200 = nfw[1]
	x2         = nfw[3]
	y2         = nfw[4]

	M200_NFW   = (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)
	e_M200_NFW = ((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*error_R200
	
	print 'R200 =',R200,'+/-', '%.2f' % error_R200,'Mpc'
	print 'M200 nfw =',M200_NFW,'+/-',e_M200_NFW


	#------------------------------------------------------
	# ------- FOR THE PLOT ------

	x=np.zeros(2,float)
	y=np.zeros(2,float)
	x[1]=10000.

	rcParams['font.family'] = 'serif'
	#rcParams['figure.figsize'] =11.,5.
	pylab.rcParams.update()
	majorFormatter = FormatStrFormatter('%.1f')
	fig = plt.figure(figsize=(8, 6))  #tamano del plot
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) #divide en 2 el eje x, en 1 el eje y y da la razon de alturas

	#asigna los sublots

	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	#grafica


	blancox=5000.
	blancoy=5000.

	name_label = sample


	ax.plot(R,shear,'ko')
	ax.plot(blancox,blancoy,'w.',label=name_label)#

	ax.legend(loc=1,frameon=False, scatterpoints = 1)
	ax.plot(x2,y2,'r--',label='NFW: c='+str('%.1f' % c)+', $R_{200}$ = '+str('%.2f' % R200)+' $\pm$ '+str('%.2f' % error_R200)+' Mpc$\,h^{-1}_{70}$, $\chi_{red}^{2} =$'+str('%.1f' % CHI_nfw)) #, c='+str(round(c)))
	#ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,-1)))+' $\pm$ '+str(int(round(ERRORDISP,-1)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.plot(X,Y,'b-',label='SIS: $\sigma$ = '+str(int(round(DISP,0)))+' $\pm$ '+str(int(round(ERRORDISP,0)))+' km/s, $\chi_{red}^{2} =$'+str('%.1f' % CHI_sis))
	ax.errorbar(R, shear, yerr=err_et, fmt=None, ecolor='k')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax.legend(loc=1,frameon=False, scatterpoints = 1)


	# axis detail
	ax.axis([RIN/1000.,ROUT/1000.,1.,5000.])
	ax.set_xscale('log', nonposy='clip')
	ax.set_yscale('log', nonposy='clip')
	ax.xaxis.set_ticks(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.set_xticklabels(np.arange(RIN/1000., ROUT/1000., 300.))
	ax.yaxis.set_ticks(np.arange(10., 200., 100.))
	ax.set_yticklabels(np.arange(10., 200., 100.))

	#label					
	ax.set_ylabel(u'$\Delta\Sigma_{\parallel} (M_{\odot}\,pc^{-2})$',fontsize=15)

	#-----------------------------

	ax2.plot(x,y,'k')
	ax2.plot(R,cero,'kx')
	ax2.errorbar(R,cero, yerr=err_ex, fmt=None, ecolor='k')

	#axis details
	ax2.axis([RIN/1000.,ROUT/1000.,-25.,30.])
	ax2.yaxis.set_ticks(np.arange(-20., 20.,15.))
	ax2.set_yticklabels(np.arange(-20., 20., 15.))
	ax2.set_xscale('log', nonposx='clip')
	#ax2.xaxis.set_ticks([0.2,0.5,1,2,3,4,5])
	#ax2.set_xticklabels([0.2,0.5,1,2,3,4,5])
	#ax2.xaxis.set_ticks([0.5,1.0,1.5,2.0,2.5])
	#ax2.set_xticklabels([0.5,1.0,1.5,2.0,2.5])
	ax2.xaxis.set_ticks([0.1,0.2,0.5,1.0,5.0])
	ax2.set_xticklabels([0.1,0.2,0.5,1.0,5.0])



	#labels
	ax2.set_ylabel(r'$\Delta\Sigma_{\times} $',fontsize=15)
	ax2.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)

	#to join the plots
	fig.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)


	plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
	plotname = 'shear_profile_'+sample+'_combined.eps'
	plt.savefig(plotname, format='eps',bbox_inches='tight')
	plt.show()		
	

		
	f1=open('profile'+sample+'_combined.cat','w')
	f1.write('# z_mean = '+str('%.2f' % zmean)+' \n')
	f1.write('# z_back = '+str('%.2f' % z_back)+' \n')
	f1.write('# odds_min = '+str('%.1f' % odds_min)+' \n')
	f1.write('# lambda_min = '+str('%.1f' %  lmin)+' \n')
	f1.write('# lambda_max = '+str('%.1f' %  lmax)+' \n')
	f1.write('# z_min = '+str('%.1f' % zmin)+' \n')
	f1.write('# z_max = '+str('%.1f' % zmax)+' \n')
	f1.write('# R,shear,err_et,cero,err_ex \n')
	profile = np.column_stack((R[:BIN],shear[:BIN],err_et[:BIN],cero[:BIN],err_ex[:BIN]))
	np.savetxt(f1,profile,fmt = ['%12.6f']*5)
	f1.close()
	
	f1=open('out_combined.cat','a')
	f1.write(str('%.2f' % zmean)+'  ')
	f1.write(str('%.2f' % z_back)+'  ')
	f1.write(str('%.1f' % odds_min)+'  ')
	f1.write(str('%.1f' %  lmin)+'  ')
	f1.write(str('%.1f' %  lmax)+'  ')
	f1.write(str('%.1f' % zmin)+'  ')
	f1.write(str('%.1f' % zmax)+'  ')
	f1.write(name_label+' '+str(int(NCUM))+' ') 
	f1.write(str('%.1f' % (lamb.mean()))+' ')
	f1.write(str(int(round(DISP,0)))+'  '+str(int(round(ERRORDISP,0)))+' ')
	f1.write(str('%.1f' % (M200_SIS/1.e12))+' '+str('%.1f' % (e_m200_SIS/1.e12))+' ')
	f1.write(str('%.1f' % CHI_sis)+' ')
	f1.write(str('%.2f' % R200)+' '+str('%.2f' % error_R200)+' ')
	f1.write(str('%.1f' % (M200_NFW/1.e12))+' '+str('%.1f' % (e_M200_NFW/1.e12))+' ')
	f1.write(str('%.1f' % CHI_nfw)+' \n')
	f1.close()	


def plot(arg):
	
	file1   = 'profileintSc_odds0_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds0_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax =  plt.subplot(321)
	ax.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])
	
	file1   = 'profileintSc_odds5_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds5_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax2 =  plt.subplot(322, sharex=ax, sharey=ax)
	ax2.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])
	
	file1   = 'profileintSc_odds6_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds6_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax2 =  plt.subplot(323, sharex=ax, sharey=ax)
	ax2.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])
	
	file1   = 'profileintSc_odds7_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds7_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax2 =  plt.subplot(324, sharex=ax, sharey=ax)
	ax2.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])
	
	file1   = 'profileintSc_odds8_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds8_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax2 =  plt.subplot(325, sharex=ax, sharey=ax)
	ax2.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])	
	
	file1   = 'profileintSc_odds9_rcut.cat'
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]
	
	file2   = 'profileSc_odds9_rcut.cat'
	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	
	ax2 =  plt.subplot(326, sharex=ax, sharey=ax)
	ax2.set_xscale("log", nonposx='clip')
	plt.plot(R1,R1*shear1,'b.')
	plt.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='b')
	plt.plot(R2,R2*shear2,'r.')
	plt.errorbar(R2,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')
	plt.axis([0.05,10.,0,115])
	
def plot_KiDS_CS82(arg):

	name_label1 = 'bin3'	
	name_label2 = 'bin4'	
	plotname = 'compare_CS82_KiDS_34.eps'
	file1   = 'profilebin3_CS82.cat'
	file2   = 'profilebin3_KiDS.cat'
	file3   = 'profilebin4_CS82.cat'
	file4   = 'profilebin4_KiDS.cat'
	
	R1      = np.loadtxt(file1)[:,0]
	shear1  = np.loadtxt(file1)[:,1]
	eshear1 = np.loadtxt(file1)[:,2]

	R2      = np.loadtxt(file2)[:,0]
	shear2  = np.loadtxt(file2)[:,1]
	eshear2 = np.loadtxt(file2)[:,2]

	R3      = np.loadtxt(file3)[:,0]
	shear3  = np.loadtxt(file3)[:,1]
	eshear3 = np.loadtxt(file3)[:,2]

	R4      = np.loadtxt(file4)[:,0]
	shear4  = np.loadtxt(file4)[:,1]
	eshear4 = np.loadtxt(file4)[:,2]

	

	rcParams['font.family'] = 'serif'
	rcParams['legend.numpoints'] = 1
	#rcParams['figure.figsize'] =11.,5.
	pylab.rcParams.update()
	majorFormatter = FormatStrFormatter('%.1f')
	fig = plt.figure(figsize=(14, 14))  #tamano del plot
	gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) #divide en 2 el eje x, en 1 el eje y y da la razon de alturas

	#asigna los sublots

	ax  = plt.subplot(gs[0,0])
	ax2 = plt.subplot(gs[1,0])
	ax3 = plt.subplot(gs[0,1], sharey=ax)
	ax4 = plt.subplot(gs[1,1], sharey=ax2)

	#grafica


	blancox=5000.
	blancoy=5000.


	ax.plot(blancox,blancoy,'w.',label=name_label1)#
	ax.plot(R2+R2*0.1,shear2,'rx', label='KiDS')
	ax.plot(R1,shear1,'ks', label='CS82')
	
	ax.errorbar(R1, shear1, yerr=eshear1, fmt=None, ecolor='k')
	ax.errorbar(R2+R2*0.1, shear2, yerr=eshear2, fmt=None, ecolor='r')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax.legend(loc=1,frameon=False, scatterpoints = 1)
	legend(numpoints=1)

	# axis detail
	ax.axis([0.1,10.,1.,1000.])
	ax.set_xscale('log', nonposy='clip')
	ax.set_yscale('log', nonposy='clip')

	#label					
	ax.set_ylabel(u'$\Delta\Sigma_{\parallel} (M_{\odot}\,pc^{-2})$',fontsize=15)

	#-----------------------------

	ax2.plot(R1,R1*shear1,'ks')
	ax2.plot(R2+R2*0.1,R2*shear2,'rx')
	ax2.errorbar(R1,R1*shear1,yerr=eshear1*R1,fmt=None,ecolor='k')
	ax2.errorbar(R2+R2*0.1,R2*shear2,yerr=eshear2*R2,fmt=None,ecolor='r')

	#axis details
	ax2.axis([0.1,10.,-25.,80.])
	ax2.set_xscale('log', nonposx='clip')

	#labels
	ax2.set_ylabel(r'$R \times \Delta\Sigma_{\parallel} (Mpc\,M_{\odot}\,pc^{-2})$',fontsize=15)
	ax2.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)
	
	###########################################
	
	ax3.plot(blancox,blancoy,'w.',label=name_label2)#
	ax3.plot(R4+R4*0.1,shear4,'rx', label='KiDS')
	ax3.plot(R3,shear3,'ks', label='CS82')
	
	ax3.errorbar(R3, shear3, yerr=eshear3, fmt=None, ecolor='k')
	ax3.errorbar(R4+R4*0.1, shear4, yerr=eshear4, fmt=None, ecolor='r')


	#legend
	matplotlib.rcParams['legend.fontsize'] = 15.
	ax3.legend(loc=1,frameon=False, scatterpoints = 1)


	# axis detail
	ax3.axis([0.1,10.,1.,1000.])
	ax3.set_xscale('log', nonposy='clip')
	ax3.set_yscale('log', nonposy='clip')


	#-----------------------------

	ax4.plot(R3,R3*shear3,'ks')
	ax4.plot(R4+R4*0.1,R4*shear4,'rx')
	ax4.errorbar(R3,R3*shear3,yerr=eshear3*R3,fmt=None,ecolor='k')
	ax4.errorbar(R4+R4*0.1,R4*shear4,yerr=eshear4*R4,fmt=None,ecolor='r')

	#axis details
	ax4.axis([0.1,10.,-25.,80.])
	ax4.set_xscale('log', nonposx='clip')

	#labels
	
	ax4.set_xlabel('r [$h^{-1}_{70}\,$Mpc]',fontsize=15)	

	#to join the plots
	fig.subplots_adjust(hspace=0,wspace =0)
	plt.setp(ax3.get_yticklabels(), visible=False)
	plt.setp(ax4.get_yticklabels(), visible=False)
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.setp(ax3.get_xticklabels(), visible=False)
	#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
	
	plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
	
	plt.savefig(plotname, format='eps',bbox_inches='tight')
	plt.show()		
	



def compare_mass(arg):
	
	m_maria   = np.array([0.83,1.3,1.3,2.9])
	e_m_maria = np.array([0.23,0.38,0.27,0.55])

	m_maria2   = np.array([1.08,1.71,1.71,3.84])
	e_m_maria2 = np.array([0.30,0.49,0.35,0.71])
	
	lamb = np.arange(15,70)
	M0_maria = 2.46
	alpha_maria = 1.18
	M200_maria = M0_maria * ((lamb/40.)**alpha_maria)
	
	M0_simet = 2.21
	alpha_simet = 1.33
	M200_simet = M0_simet * ((lamb/40.)**alpha_simet)

	M0_melchior = 2.21
	alpha_melchior = 1.12
	M200_melchior = M0_melchior * ((lamb/40.)**alpha_melchior)

	M0_oguri = 2.53
	alpha_oguri = 1.44
	M200_oguri = M0_oguri * ((lamb/40.)**alpha_oguri)

	
	out = np.loadtxt('out.cat', comments='#', dtype ='str')
	
	M200_nfw = (out[-4:,-3].astype(float)*0.7*1.e12)/1.e14
	eM200_nfw = (out[-4:,-2].astype(float)*0.7*1.e12)/1.e14
	l = out[-4:,9].astype(float)
	
	
	plt.plot(lamb,M200_maria,'k',label = 'Pereira et al 2018')
	#plt.plot(lamb,M200_simet,'r',label = 'Simet et al 2017')
	#plt.plot(lamb,M200_melchior,'b',label = 'Melchior et al 2017')
	#plt.plot(lamb,M200_oguri,'g',label = 'Oguri et al 2014')
	plt.plot(l,M200_nfw,'ko')
	plt.plot(l,m_maria,'ro')
	plt.plot(l,m_maria2,'bo')
	plt.errorbar(l,M200_nfw,yerr=eM200_nfw,fmt=None,ecolor='k')
	plt.errorbar(l,m_maria,yerr=e_m_maria,fmt=None,ecolor='r')
	plt.errorbar(l,m_maria2,yerr=e_m_maria2,fmt=None,ecolor='b')
	plt.legend(loc=2)
	plt.xlabel('$\lambda$',fontsize = 18)
	plt.ylabel('$M_{200} [10^{14} h^{-1} M_\odot]$',fontsize = 18)
	plt.show()


def red_blue(Mr1,Mg1,Mr2,Mg2,index):

	L1=10**(-0.4*Mg1)
	L2=10**(-0.4*Mg2)
	Mg=-2.5*np.log10(L1+L2)

	L1=10**(-0.4*Mr1)
	L2=10**(-0.4*Mr2)
	Mr=-2.5*np.log10(L1+L2)
	
	color=Mg-Mr
	MR=Mr[index]
	COLOR=color[index]
	mmag=MR>-26.
	mag_bins=np.array([np.percentile(MR[mmag],10),np.percentile(MR[mmag],20),
	                  np.percentile(MR[mmag],30),np.percentile(MR[mmag],40),
	                  np.percentile(MR[mmag],50),np.percentile(MR[mmag],60),
	                  np.percentile(MR[mmag],70),np.percentile(MR[mmag],80),
	                  np.percentile(MR[mmag],90),np.percentile(MR[mmag],100)])

	mag_mean=np.zeros(len(mag_bins)-1)
	color_mean=np.zeros(len(mag_bins)-1)
	
	for j in range(len(mag_mean)):
		mask=(MR>mag_bins[j])*(MR<mag_bins[j+1])
		mag_mean[j]=0.5*(mag_bins[j]+mag_bins[j+1])
		color_mean[j]=np.median(COLOR[mask])
	
	
	popt, pcov = curve_fit(lambda x,m,n: x*m+n, mag_mean,color_mean)
	m,n = popt
	
	red=np.zeros(len(Mg))
	RED=np.zeros(len(COLOR))
	
	mred=color>(Mr*m+n)
	mred2=COLOR>(MR*m+n)
	red[mred]=1
	RED[mred2]=1
	
	# rcParams['figure.figsize'] =10.,10.

	# fig = plt.figure(figsize=(7.5,7.5))
	# fig.subplots_adjust(hspace=0,wspace=0)

	# gs = GridSpec(4,4)

	# ax_joint = fig.add_subplot(gs[1:4,0:3])
	# ax_marg_x = fig.add_subplot(gs[0,0:3])
	# ax_marg_y = fig.add_subplot(gs[1:4,3])
	
	# ax_joint.plot(MR[mred2],COLOR[mred2],'r.',alpha = 0.8, label = 'Red pairs')
	# ax_joint.plot(MR[~mred2],COLOR[~mred2],'b.', alpha = 0.8, label = 'Blue pairs')
	# ax_joint.plot(mag_mean,color_mean,'ko')
	# ax_joint.plot(MR,m*MR+n,'k')
	# ax_joint.set_ylabel(r'$M_g - M_r$',fontsize=16)
	# ax_joint.set_xlabel(r'$M_r$',fontsize=16)
	# ax_joint.xaxis.set_ticks(np.arange(-23.5,-19.,0.5))
	# ax_joint.set_xticklabels(np.arange(-23.5,-19.,0.5))
	
	
	# ax_marg_x.hist(MR,histtype='step',stacked=True,fill=False,color='k')
	# ax_marg_x.axvline(np.median(MR))
	# ax_marg_x.yaxis.set_ticks(np.arange(10,120,20))
	# ax_marg_x.set_yticklabels(np.arange(10,120,20))
	# ax_marg_x.set_ylabel(r'$N$',fontsize=16)
	# ax_marg_x.text(-23.4,90.,'Higher-luminosity', fontsize=14.)
	# ax_marg_x.text(-21.,90.,'Lower-luminosity', fontsize=14.)
	
	# ax_marg_y.hist(COLOR,orientation="horizontal",histtype='step',stacked=True,fill=False,color='k')
	# ax_marg_y.xaxis.set_ticks(np.arange(20,120,20))
	# ax_marg_y.set_xticklabels(np.arange(20,120,20))
	# ax_marg_y.set_xlabel(r'$N$',fontsize=16)
	# ax_marg_y.axis([0,110,0.,1.4])
	

	# plt.setp(ax_marg_x.get_xticklabels(), visible=False)
	# plt.setp(ax_marg_y.get_yticklabels(), visible=False)
	# plt.savefig('color_mag.pdf', format='pdf',bbox_inches='tight')
	# plt.show()
	
	
	
	return red,RED
