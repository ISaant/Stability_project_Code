#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:09:14 2023

@author: sflores
"""

from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model
import matplotlib.pyplot as plt

plt.close('all')
# Set random seed, for consistency generating simulated data
set_random_seed(21)

# Simulate example power spectra
freqs1, powers1 = gen_power_spectrum([3, 40], [1, 1],
                                     [[10, 0.2, 1.25], [30, 0.15, 2]])
freqs2, powers2 = gen_power_spectrum([1, 150], [1, 125, 1.25],
                                     [[8, 0.15, 1.], [30, 0.1, 2]])

# Initialize power spectrum model objects and fit the power spectra
fm1 = FOOOF(min_peak_height=0.05, aperiodic_mode='knee', verbose=False)
fm2 = FOOOF(min_peak_height=0.05, aperiodic_mode='knee', verbose=False)
fm1.fit(freqs1, powers1)
fm2.fit(freqs2, powers2)

# Plot one of the example power spectra
plot_spectrum(freqs2, powers2, log_powers=True,
              color='black', label='Original Spectrum')

plt.show()

# %% Visualizing Power Spectrum Models

# Plot an example power spectrum, with a model fit
fm1.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})
fm2.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'},plt_log=True)
plt.show()

# %% 
from fooof.utils.download import load_fooof_data
# Download example data files needed for this example
freqs = load_fooof_data('freqs.npy', folder='data')
spectrum = load_fooof_data('spectrum.npy', folder='data')

# Initialize a FOOOF object
# fm = FOOOF(min_peak_height=0.05, aperiodic_mode='fixed')
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [2, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
# fm.report(freqs, spectrum, freq_range)

# Alternatively, just fit the model with FOOOF.fit() (without printing anything)
fm.fit(freqs, spectrum, freq_range)

# After fitting, plotting and parameter fitting can be called independently:
fm.print_results()
fm.plot()

# Aperiodic parameters
print('Aperiodic parameters: \n', fm.aperiodic_params_, '\n')
offset,exponet=fm.aperiodic_params_
# Peak parameters
print('Peak parameters: \n', fm.peak_params_, '\n')

# Goodness of fit measures
print('Goodness of fit:')
print(' Error - ', fm.error_)
print(' R^2   - ', fm.r_squared_, '\n')

# Check how many peaks were fit
print('Number of fit peaks: \n', fm.n_peaks_)
plt.show()

# Extract a model parameter with `get_params`
err = fm.get_params('error')

# Extract parameters, indicating sub-selections of parameters
exp = fm.get_params('aperiodic_params', 'exponent')
cfs = fm.get_params('peak_params', 'CF')

# Print out a custom parameter report
template = ("With an error level of {error:1.2f}, FOOOF fit an exponent "
            "of {exponent:1.2f} and peaks of {cfs:s} Hz.")
print(template.format(error=err, exponent=exp,
                      cfs=' & '.join(map(str, [round(cf, 2) for cf in cfs]))))

fres = fm.get_results() #Aperiodic, Periodic, Error, R^2, Real Gaussian values
ap_params, peak_params, r_squared, fit_error, gauss_params = fm.get_results()