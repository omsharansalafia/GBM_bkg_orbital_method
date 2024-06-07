# GBM_bkg_orbital_method
Code to estimate the Fermi/GBM background with the "orbital" method

# Prerequisites
- this code is written in python 3.x and uses the following packages: numpy, matplotlib, scipy, astropy, emcee
- the GBM daily data needed to reproduce the results in Ravasio et al. 2024 consists of the following files: glg_cspec_BGO0_221007_v00.pha, glg_cspec_BGO0_221008_v00.pha, glg_cspec_BGO0_221009_v00.pha, glg_cspec_BGO0_221010_v00.pha, glg_cspec_BGO0_221011_v00.pha. These files must be placed in the root directory of the repository and can be downloaded from the GBM daily data catalog: https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermigdays.html

# Usage
- the average orbit duration is obtained by maximising the cross-correlation of the daily data across different days. This is done in maximise_cross_correlation.py
- the data at +/-30 orbits is assumed to be background and is fitted assuming a 4-th order polynomial shape in each of the 128 channels. This is done in fit_bkg_with_orbital_method.py 
