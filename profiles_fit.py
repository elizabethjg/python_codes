import sys
import numpy as np
from cosmolopy import *
from scipy.optimize import curve_fit
from scipy import integrate

cvel=299792458;   # Speed of light (m.s-1)
G= 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc= 3.085678e16; # 1 pc (m)
Msun=1.989e30 # Solar mass (kg)


def disp_sis(M200,H):
	'''
	
	Returns the velocity dispersion according to a M200 mass
	assuming an SIS model
	------------------------------------------------------------------
	INPUT:
	M200         (float or array of floats) M_200 mass in solar masses
	H            (float or array of floats) Hubble constant computed
	             at the redshift of the halo
	------------------------------------------------------------------
	OUTPUT:
	disp         (float or array of floats) Velocity dispersion
	
	'''
	return (((M200*((50**0.5)*G*H*Msun))/2.)**(1./3.))/1.e3
	

def r200_nfw(M200,roc_mpc):	
	'''
	
	Returns the R_200
	------------------------------------------------------------------
	INPUT:
	M200         (float or array of floats) M_200 mass in solar masses
	roc_mpc      (float or array of floats) Critical density at the z 
	             of the halo in units of kg/Mpc**3
	------------------------------------------------------------------
	OUTPUT:
	R_200         (float or array of floats) 
	
	'''

	return ((M200*(3.0*Msun))/(800.0*np.pi*roc_mpc))**(1./3.)



def chi_red(ajuste,data,err,gl):
	'''
	Reduced chi**2
	------------------------------------------------------------------
	INPUT:
	ajuste       (float or array of floats) fitted value/s
	data         (float or array of floats) data used for fitting
	err          (float or array of floats) error in data
	gl           (float) grade of freedom (number of fitted variables)
	------------------------------------------------------------------
	OUTPUT:
	chi          (float) Reduced chi**2 	
	'''
		
	BIN=len(data)
	chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
	return chi



def sis_profile_sigma(R,sigma):
	Rm=R*1.e6*pc
	return (((sigma*1.e3)**2)/(2.*G*Rm))*(pc**2/Msun)



def SIS_stack_fit(R,D_Sigma,err):
	
	# R en Mpc, D_Sigma M_Sun/pc2
	

	sigma,err_sigma_cuad=curve_fit(sis_profile_sigma,R,D_Sigma,sigma=err,absolute_sigma=True)
	
	ajuste=sis_profile_sigma(R,sigma)
	
	chired=chi_red(ajuste,D_Sigma,err,1)	
	
	xplot=np.arange(0.001,R.max()+1.,0.001)
	yplot=sis_profile_sigma(xplot,sigma)
	
	return sigma[0],np.sqrt(err_sigma_cuad)[0][0],chired,xplot,yplot


def NFW_profile_sigma(datos,R200):
	
	R, roc_mpc, z, h = datos
	#calculo de c usando la relacion de Duffy et al 2008
	
	M=((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*h
	c=5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
	# c = 5.0
	####################################################
	
	deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
	x=(R*c)/R200
	m1=x< 1.0
	atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
	jota=np.zeros(len(x))
	jota[m1]=(4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
		+ (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
		+ (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))
	m2=x> 1.0     
	atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
	jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
		+ (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
		+ (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
	m3=(x == 1.0)
	jota[m3]=2.0*np.log(0.5)+5.0/3.0
	rs_m=(R200*1.e6*pc)/c
	kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
	return kapak*jota

def NFW_profile_sigma_c(datos,R200,c):
	
	R, roc_mpc, z = datos
	
	deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
	x=(R*c)/R200
	m1=x< 1.0
	atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
	jota=np.zeros(len(x))
	jota[m1]=(4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
		+ (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
		+ (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))
	m2=x> 1.0     
	atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
	jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
		+ (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
		+ (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
	m3=(x == 1.0)
	jota[m3]=2.0*np.log(0.5)+5.0/3.0
	rs_m=(R200*1.e6*pc)/c
	kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
	return kapak*jota

	
def NFW_stack_fit(R,D_Sigma,err,z,roc,fitc = False, h = 0.7):
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)
	roc_mpc=roc*((pc*1.0e6)**3.0)

	if fitc:
		NFW_out = curve_fit(NFW_profile_sigma_c,(R, roc_mpc,z),D_Sigma,sigma=err,absolute_sigma=True)
		pcov    = NFW_out[1]
		perr    = np.sqrt(np.diag(pcov))
		e_R200  = perr[0]
		e_c     = perr[1]
		R200    = NFW_out[0][0]
		c       = NFW_out[0][1]
		ajuste  = NFW_profile_sigma_c((R, roc_mpc,z),R200,c)
		chired  = chi_red(ajuste,D_Sigma,err,2)	
	
		xplot   = np.arange(0.001,R.max()+1.,0.001)
		yplot   = NFW_profile_sigma_c((xplot,roc_mpc,z),R200,c)

	else:
		NFW_out = curve_fit(NFW_profile_sigma,(R, roc_mpc,z,h),D_Sigma,sigma=err,absolute_sigma=True)
		e_R200  = np.sqrt(NFW_out[1][0][0])
		R200    = NFW_out[0][0]
		
		ajuste  = NFW_profile_sigma((R, roc_mpc,z,h),R200)
		
		chired  = chi_red(ajuste,D_Sigma,err,1)	
	
		xplot   = np.arange(0.001,R.max()+1.,0.001)
		yplot   = NFW_profile_sigma((xplot,roc_mpc,z,h),R200)
	
		#calculo de c usando la relacion de Duffy et al 2008
		M   = ((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*0.7
		c   = 5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
		e_c = 0.
		####################################################
	
	return R200,e_R200,chired,xplot,yplot,c,e_c
	
def SIGMA_nfw(datos,R200):		
	'''
	Surface mass density for NFW (Eq. 11 - Wright and Brainerd 2000)
	------------------------------------------------------------------
	INPUT:
	datos        (list or tupple) contains [R,roc_mpc,z]
	             R        (float array) distance to the centre in Mpc
	             roc_mpc  (float) Critical density at the z of the halo in units of kg/Mpc**3
	             z        (float) Redshift of the halo
	             h        (float) Incertanty factor in constant Huble
	R200         (float)  R_200 in Mpc
	------------------------------------------------------------------
	OUTPUT:
	Sigma(R)     (float array) Surface mass density in units of M_Sun/pc2
	'''		
	R, roc_mpc, z, h = datos
	
	if not isinstance(R, (np.ndarray)):
		R = np.array([R])
	
	#calculo de c usando la relacion de Duffy et al 2008
	M=((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*h
	c=5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
	####################################################
	
	deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
	c = c.astype(float128)
	x=(R*c)/R200
	m1 = x < 1.0
	m2 = x > 1.0
	m3 = x == 1.0

	jota  = np.zeros(len(x))
	atanh = np.arctanh(np.sqrt((1.0-x[m1])/(1.0+x[m1])))
	jota[m1] = (1./(x[m1]**2-1.))*(1.-(2./np.sqrt(1.-x[m1]**2))*atanh) 

	atan = np.arctan(np.sqrt((x[m2]-1.0)/(1.0+x[m2])))
	jota[m2] = (1./(x[m2]**2-1.))*(1.-(2./np.sqrt(x[m2]**2 - 1.))*atan) 

	jota[m3] = 1./3.
					
	rs_m=(R200*1.e6*pc)/c
	
	kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
	return kapak*jota

	
def shear_map(x,y,e,theta,npix):
	stepx=(x.max()-x.min())/npix
	stepy=(y.max()-y.min())/npix
	xbin=np.zeros(npix**2,float)
	ybin=np.zeros(npix**2,float)
	ex=np.zeros(npix**2,float)
	ey=np.zeros(npix**2,float)
	ngx=np.zeros(npix**2,float)
	#~ plt.plot(x,y,'k.')
	inx=x.min()
	
	ind=0
	print len(ex)
	for j in range(npix):
		#~ plt.plot(x,y,'k.')
		maskx=(x>inx)*(x<(inx+stepx))
		#~ plt.plot(x[maskx],y[maskx],'r.')
		iny=y.min()
		for i in range(npix):
			masky=(y[maskx]>iny)*(y[maskx]<(iny+stepy))
			#~ plt.plot(x[maskx][masky],y[maskx][masky],'b.')
			#~ print ind,len(e[maskx][masky]),iny,iny+stepy
			ex[ind]=e[maskx][masky].mean()*np.cos(theta[maskx][masky].mean())
			ey[ind]=e[maskx][masky].mean()*np.sin(theta[maskx][masky].mean())
			xbin[ind]=x[maskx][masky].mean()
			ybin[ind]=y[maskx][masky].mean()
			ngx[ind]=len(y[maskx][masky])
			ind=ind+1
			iny=iny+stepy
		inx=inx+stepx	
		#~ plt.show()
	return xbin,ybin,ex,ey,ngx
	
def shear_map2(x,y,e1,e2,npix):
	stepx=(x.max()-x.min())/npix
	stepy=(y.max()-y.min())/npix
	xbin=np.zeros(npix**2,float)
	ybin=np.zeros(npix**2,float)
	ex=np.zeros(npix**2,float)
	ey=np.zeros(npix**2,float)
	ngx=np.zeros(npix**2,float)
	#~ plt.plot(x,y,'k.')
	inx=x.min()
	
	ind=0
	print len(ex)
	for j in range(npix):
		#~ plt.plot(x,y,'k.')
		maskx=(x>inx)*(x<(inx+stepx))
		#~ plt.plot(x[maskx],y[maskx],'r.')
		iny=y.min()
		for i in range(npix):
			masky=(y[maskx]>iny)*(y[maskx]<(iny+stepy))
			#~ plt.plot(x[maskx][masky],y[maskx][masky],'b.')
			#~ print ind,len(e[maskx][masky]),iny,iny+stepy
			ex[ind]=e1[maskx][masky].mean()
			ey[ind]=e2[maskx][masky].mean()
			xbin[ind]=x[maskx][masky].mean()
			ybin[ind]=y[maskx][masky].mean()
			ngx[ind]=len(y[maskx][masky])
			ind=ind+1
			iny=iny+stepy
		inx=inx+stepx	
		#~ plt.show()
	return xbin,ybin,ex,ey,ngx

'''
ELLIPTICAL EQUATIONS FOR A SIS PROFILE
'''


def delta_fi(fi):
	return 1./np.sqrt(np.cos(fi)**2+(f**2)*np.sin(fi)**2)


def esis_profile_sigma(RDfi,f,disp):
	R,fi = RDfi
	Rm=R*1.e6*pc
	R0 = ((disp*1.e3)**2)/G
	b = (Rm/R0)*np.sqrt((f**2)*np.cos(fi)**2+np.sin(fi)**2)
	D_Sigma = (np.sqrt(f)/(2.*b))*(pc**2/Msun)
	Rout = R*np.sqrt((f**2)*np.cos(fi)**2+np.sin(fi)**2)
	return D_Sigma
	
		
def e_SIS_stack_fit(R,fi,D_Sigma,err):	

	e_SIS_out=curve_fit(esis_profile_sigma2,(R,fi),D_Sigma,sigma=err,absolute_sigma=True)
	pcov=e_SIS_out[1]
	perr = np.sqrt(np.diag(pcov))
	e_f=perr[0]
	e_disp=perr[1]
	f=e_SIS_out[0][0]
	disp=e_SIS_out[0][1]		




def esis_profile_sigma_mod_per(R,fi,f,disp=250.):
	Rm = R*1.e6*pc
	R0 = ((disp*1.e3)**2)/G
	x2 = lambda fi: 1./np.sqrt((f**2)*np.cos(fi)**2+np.sin(fi)**2)
	integral1 = integrate.quad(x2, 0.25*np.pi, 0.75*np.pi)[0]
	integral2 = integrate.quad(x2, 1.25*np.pi, 1.75*np.pi)[0]
	D_Sigma = (R0/Rm)*(np.sqrt(f)/np.pi)*(integral1+integral2)
	Rout = R
	return Rout,D_Sigma	
	
	
def esis_profile_sigma_mod_par(R,fi,f,disp=250.):
	Rm = R*1.e6*pc
	R0 = ((disp*1.e3)**2)/G
	x2 = lambda fi: 1./np.sqrt((f**2)*np.cos(fi)**2+np.sin(fi)**2)
	integral1 = integrate.quad(x2, 0.*np.pi, 0.25*np.pi)[0]
	integral2 = integrate.quad(x2, 0.75*np.pi, 1.25*np.pi)[0]
	integral3 = integrate.quad(x2, 1.75*np.pi, 2.*np.pi)[0]
	D_Sigma = (R0/Rm)*(np.sqrt(f)/np.pi)*(integral1+integral2+integral3)
	Rout = R
	return Rout,D_Sigma	
