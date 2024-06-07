import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

# which BGO detector to use
bgo = 'b1'

# the trigger time
trigtime = Time('2022-10-09 13:16:59.988')


T = []
CR = []

# open count rate data
for day in ['07','08','09','10','11']:
    
    fname = 'glg_cspec_{0}_2210{1}_v00.pha'.format(bgo,day)
    hdu = fits.open(fname)
    
    reftime = Time(hdu[0].header['MJDREFI'],format='mjd')
        
    counts = hdu[2].data['COUNTS']
        
    t = hdu[2].data['TIME']*u.s+(reftime-trigtime).to('s')
    
    dt = hdu[2].data['EXPOSURE']*u.s
    q = hdu[2].data['QUALITY']
    
    good = (dt>0) & (q==0) & ((t<-10.*u.s) | (t>1500.*u.s))  
    
    T.append(t[good])
    CR.append(np.sum(counts,axis=1)[good]/dt[good])


T = np.concatenate(T)
CR = np.concatenate(CR)

cri = interp1d(T,CR)

crm = np.trapz(CR,T)/(T.max()-T.min())
crv = np.trapz((CR-crm)**2,T)/(T.max()-T.min())

revtimes = np.linspace(93.,97.,1000)

cc = np.zeros(len(revtimes))

for i,r in enumerate(revtimes):
    tmax = T.max()-30.*revtimes.max()*60.*u.s
    t = np.linspace(T.min(),tmax,1000)
    cc[i] = (np.trapz((cri(t)/u.s-crm)*(cri(t+30.*r*60.*u.s)/u.s-crm),t)/(t.max()-t.min()))/crv

plt.plot(revtimes,savgol_filter(cc,20,2),'-r')
plt.xlabel(r'Average orbit duration [min]')
plt.ylabel(r'Cross correlation')
plt.xlim(revtimes.min(),revtimes.max())
plt.ylim(-0.1,1.1)
plt.tick_params(which='both',direction='in',top=True,right=True)
plt.show()
    
    
    
