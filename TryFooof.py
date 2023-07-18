#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:17:32 2023

@author: isaac
"""

# Import the FOOOF object
from fooof import FOOOF

# Import a utility to download and load example data
from fooof.utils.download import load_fooof_data

# Download example data files needed for this example
freqss = load_fooof_data('freqs_2.npy', folder='data')
spectrum = load_fooof_data('spectrum_2.npy', folder='data')*10000000000000000000000
columnss=np.where(np.logical_and(freqss>=1,freqss<=40))
plt.plot(freqss[columnss],spectrum[columnss])
#   These settings will be more fully described later in the tutorials
fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.15)
fm.add_data(freqss[columnss], spectrum[columnss], [1, 40])
# Fit the power spectrum model
fm.fit(freqss, spectrum, [1, 40])
exp = fm.get_params('aperiodic_params', 'exponent')
offset = fm.get_params('aperiodic_params', 'offset')
arythmic=offset-np.log10(freqss[columnss]**exp)
plt.plot(freqss[columnss],fm.power_spectrum)
plt.plot(freqss[columnss],fm._ap_fit)
plt.plot(freqss[columnss],10**arythmic)
plt.plot(freqss[columnss],fm.power_spectrum-fm._ap_fit)
plt.plot(freqss[columnss],spectrum[columnss]-10**arythmic)

#%%
plt.figure()
Data=copy.copy(restStateCropped[:,2:,:])
roi=0
sub=0
fm = FOOOF(max_n_peaks=6, aperiodic_mode='fixed',min_peak_height=0.15,verbose=False)
fm.add_data(freqsFooof, restState[sub,columnsfooof,roi],inBetween) #freqs[0]<inBetween[:]<freqs[1]
fm.fit(freqsFooof, restState[sub,columnsfooof,roi], inBetween)
whitened=fm.power_spectrum-fm._ap_fit
exp = fm.get_params('aperiodic_params', 'exponent')
offset = fm.get_params('aperiodic_params', 'offset')
arythmic=offset-np.log10(freqs[columns[2:]]**exp)
exp = parameters[roi][sub][0]
offset = parameters[roi][sub][1]
arythmic2=offset-np.log10(freqs[columns[2:]]**exp)
a=Data[sub,:,roi]-10**arythmic
plt.plot(freqs[columnsfooof],fm.power_spectrum)
plt.plot(freqss[columnsfooof],fm._ap_fit)
plt.plot(freqs[columns[2:]],Data[sub,:,roi])
plt.plot(freqs[columns[2:]],10**arythmic)
plt.plot(freqs[columns[2:]],arythmic2)

plt.figure()
plt.plot(freqs[columnsfooof],fm.power_spectrum-fm._ap_fit)
plt.plot(freqs[columns[2:]],10**(np.log10(Data[sub,:,roi])-arythmic2))
