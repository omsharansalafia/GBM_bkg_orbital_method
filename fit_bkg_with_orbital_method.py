import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import emcee
from multiprocessing import Pool

def poly(x,a,order=4):
    return np.sum(a[None,:]*((x[:,None])**np.arange(0,order+1)),axis=1)

def poiss_loglike(nobs,nexp):
    return np.sum(nobs*np.nan_to_num(np.log(nexp))-nexp)

revolution_time = 95.04*60.*u.s #revolution time in seconds

bgo = 'b0'

trigtime = Time('2022-10-09 13:16:59.988')

t0pre = 100*u.s
t1pre = 280*u.s

t0post = 320.*u.s
t1post = 500*u.s

tm = np.linspace(0.,600.,1000)*u.s

nsamples = 1000
bkgm = np.zeros([2,nsamples,128,len(tm)])

hdu = fits.open("glg_cspec_{0}_221009_v00.pha".format(bgo))
counts_src = hdu[2].data['COUNTS']
reftime = Time(hdu[0].header['MJDREFI'],format='mjd')
t_src = hdu[2].data['TIME']*u.s+(reftime-trigtime).to('s')
dt_src = hdu[2].data['EXPOSURE']


for io,orbit in enumerate([-30,30]):
    
    
    day = '{0:02d}'.format((trigtime+orbit*revolution_time).datetime.day)
    fname = 'glg_cspec_{0}_2210{1}_v00.pha'.format(bgo,day)
    hdu = fits.open(fname)
    
    reftime = Time(hdu[0].header['MJDREFI'],format='mjd')
    
    chan = hdu[1].data['CHANNEL']
    emin = hdu[1].data['E_MIN']
    emax = hdu[1].data['E_MAX']
    
    counts = hdu[2].data['COUNTS']
        
    t = hdu[2].data['TIME']*u.s+(reftime-trigtime).to('s')-orbit*revolution_time
    
    dt = hdu[2].data['EXPOSURE']
    q = hdu[2].data['QUALITY']
    
    i0 = np.searchsorted(t.to('s').value,t0pre.to('s').value)
    
    sel = ((t>=t0pre)&(t<=t1pre))|((t>=t0post)&(t<=t1post))
    sel2 = ((t>=t0pre)&(t<=t1post))
    
    
    for j in range(len(chan)):
        print(j)
        cr0 = counts[i0,j]/dt[i0]
        initparams = np.array([np.log(np.maximum(cr0,0.1)),1.250e-04,-4.280e-10,-6.085e-11,3.434e-14])
        stat = lambda a: -0.5*poiss_loglike(counts[sel,j],np.exp(poly((t[sel]-t0pre).to('s').value,a))*dt[sel])
        sol = minimize(stat,initparams,method='Nelder-Mead')
        print(sol)
        
        def loglike(a): 
            return poiss_loglike(counts[sel,j],np.exp(poly((t[sel]-t0pre).to('s').value,a))*dt[sel])
        
        ndim = len(sol.x)
        
        nwalkers = ndim*4
        nsteps = 1000
            
        # the initial positions of the walkers are uniformly distributed within the bounds
        p0s = (np.array(sol.x + 1e-14).reshape([len(sol.x),1])*(1. + np.random.normal(loc=0.,scale=1e-2,size=[ndim,nwalkers]))).T
        
        # set up the backend
        filename = "chains_{2}/channel_{0:d}_{1:d}.h5".format(j,io,bgo)
        backend = emcee.backends.HDFBackend(filename)
        
        # initialize the sampler
        with Pool(6) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,ndim,loglike,backend=backend,pool=pool)
            if sampler.iteration>0:
                sampler.run_mcmc(None,nsteps-sampler.iteration,progress=True)
            else:
                sampler.run_mcmc(p0s,nsteps,progress=True,skip_initial_state_check=False)
        
        # use samples
        x = sampler.get_chain(flat=True,discard=nsteps-nsamples//nwalkers)
        print(x.shape)
    
        for k in range(nsamples):
            bkgm[io,k,j,:] = np.exp(poly((tm-t0pre).to('s').value,x[k]))
        
       
    plt.figure('lc{0:d}'.format(io))
    plt.step(t_src,np.sum(counts_src,axis=1)/dt_src,'-k',alpha=0.5)
    plt.step(t,np.sum(counts,axis=1)/dt,'-r')
    plt.fill_between(tm.to('s').value,np.percentile(np.sum(bkgm[io],axis=1),5.,axis=0),np.percentile(np.sum(bkgm[io],axis=1),95.,axis=0),color='cyan',alpha=0.3)
    plt.plot(tm.to('s').value,np.median(np.sum(bkgm[io],axis=1),axis=0),'-b')
    
    t0 = 280.
    t1 = 300.
    
    bkg_cumul = interp1d(tm.value,cumtrapz(bkgm[io],tm.to('s').value,initial=0.,axis=2),axis=2)
    
    bkg_spec_1 = (bkg_cumul(t1)-bkg_cumul(t0))/(t1-t0)/(emax-emin)[None,:]
    
    t0 = 300.
    t1 = 320.
    
    bkg_spec_2 = (bkg_cumul(t1)-bkg_cumul(t0))/(t1-t0)/(emax-emin)[None,:]
    
    bkg_spec_1_16,bkg_spec_1_50,bkg_spec_1_84 = np.percentile(bkg_spec_1,[16.,50.,84.],axis=0)
    bkg_spec_2_16,bkg_spec_2_50,bkg_spec_2_84 = np.percentile(bkg_spec_2,[16.,50.,84.],axis=0)
    
    np.savetxt('GRB221009A_BGO{1}_bkg_fit_280-300_{0:+d}.txt'.format(orbit,bgo[1]),np.vstack([emin,emax,bkg_spec_1_50,0.5*(bkg_spec_1_84-bkg_spec_1_16)]).T,header='E_min E_max bkg bkg_err')
    np.savetxt('GRB221009A_BGO{1}_bkg_fit_300-320_{0:+d}.txt'.format(orbit,bgo[1]),np.vstack([emin,emax,bkg_spec_2_50,0.5*(bkg_spec_2_84-bkg_spec_2_16)]).T,header='E_min E_max bkg bkg_err')
    
    plt.figure('spec1')
    plt.errorbar(0.5*(emin+emax),bkg_spec_1_50,xerr=0.5*(emax-emin),yerr=[bkg_spec_1_50-bkg_spec_1_16,bkg_spec_1_84-bkg_spec_1_50],marker='.',ls='None',ms=0.,label=orbit)
    
    plt.figure('spec2')
    plt.errorbar(0.5*(emin+emax),bkg_spec_2_50,xerr=0.5*(emax-emin),yerr=[bkg_spec_2_50-bkg_spec_2_16,bkg_spec_2_84-bkg_spec_2_50],marker='.',ls='None',ms=0.,label=orbit)

t0 = 280.
t1 = 300.

bkg_cumul = interp1d(tm.value,cumtrapz(np.mean(bkgm,axis=0),tm.to('s').value,initial=0.,axis=2),axis=2)
bkg_spec_1 = (bkg_cumul(t1)-bkg_cumul(t0))/(t1-t0)/(emax-emin)[None,:]

t0 = 300.
t1 = 320.

bkg_spec_2 = (bkg_cumul(t1)-bkg_cumul(t0))/(t1-t0)/(emax-emin)[None,:]

bkg_spec_1_16,bkg_spec_1_50,bkg_spec_1_84 = np.percentile(bkg_spec_1,[16.,50.,84.],axis=0)
bkg_spec_2_16,bkg_spec_2_50,bkg_spec_2_84 = np.percentile(bkg_spec_2,[16.,50.,84.],axis=0)

np.savetxt('GRB221009A_BGO{0}_bkg_fit_280-300_avg.txt'.format(bgo[1]),np.vstack([emin,emax,bkg_spec_1_50,0.5*(bkg_spec_1_84-bkg_spec_1_16)]).T,header='E_min E_max bkg bkg_err')
np.savetxt('GRB221009A_BGO{0}_bkg_fit_300-320_avg.txt'.format(bgo[1]),np.vstack([emin,emax,bkg_spec_2_50,0.5*(bkg_spec_2_84-bkg_spec_2_16)]).T,header='E_min E_max bkg bkg_err')

plt.figure('spec1')
plt.errorbar(0.5*(emin+emax),bkg_spec_1_50,xerr=0.5*(emax-emin),yerr=[bkg_spec_1_50-bkg_spec_1_16,bkg_spec_1_84-bkg_spec_1_50],marker='.',ls='None',ms=0.,label='avg')
    
plt.figure('spec2')
plt.errorbar(0.5*(emin+emax),bkg_spec_2_50,xerr=0.5*(emax-emin),yerr=[bkg_spec_2_50-bkg_spec_2_16,bkg_spec_2_84-bkg_spec_2_50],marker='.',ls='None',ms=0.,label='avg')

plt.show()
