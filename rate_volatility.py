#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:11:14 2021

@author: jtobin
"""

from auth_quandl import authenticate_quandl
import rate_volatility_plots as rvp
import rate_volatility_functions as rvf
import os
import datetime as dt
from arch import arch_model
import quandl


# local location
local_dir = os.path.expanduser('~/Documents/projects/rate_volatility')
os.chdir(local_dir)

# load in module with required functions

# load in authenticate quandl code; authenticate
quandl = authenticate_quandl(quandl)

# make directory for plots. if exists, clear it out
if not os.path.exists('Plots'):
    os.mkdir('Plots')
else:
    [os.remove('Plots/' + fn) for fn in os.listdir('Plots')]

# empty list for plot paths for later combination into a PDF
plot_fs = []

# load in data
data = rvf.load_rates_data(False)
rates = data['data']
labels = data['labels']
label_vals = {k: v for k, v in zip(labels, rates.columns)}

####
# make basic plots of current yield curve, curve over time, and historical
# yield volatility dynamics
####

# plot latest curve
f = 'Plots/latest_curve.png'
rvp.plot_latest_curve(rates, labels, f)
plot_fs.append(f)

# yield curve surface plot
yc_date = dt.datetime(2006, 2, 9)
f = 'Plots/yc_surface.png'
rvp.plot_yc_surface(rates, yc_date, labels, f)
plot_fs.append(f)

# 10YT volatility plots
rate_dfs = {}
for t in ['3 MO', '2 YR', '10 YR', '30 YR']:
    f = 'Plots/' + t + '_stats.png'
    rate_dfs[t] = rvp.plot_yld_vol(rates, label_vals, t, f)
    plot_fs.append(f)

####
# determine degree of mean reverseion present in the series using Hurst exp.
####

# hurst
t10 = rate_dfs['10 YR']
rvf.hurst(t10['10 YR'].values)
rvf.hurst(t10['Diff'].dropna().values)
rvf.hurst(t10['LogDiff'].dropna().values)

# > 10Y Yield Hurst is close to 0, suggesting a mean reverting series. Nice!

####
# make plot of changes in yield and serial correlation over time
####

# create standardized version of log diffs
subset = t10['LogDiff'].dropna()
standardized = subset.sub(subset.mean()) ** 2

# make plots
f_l = 'Plots/Log Diff Autocorr.png'
rvp.plot_corr(subset, 'Log Diff Autocorr', f_l)
f_sl = 'Plots/Standardized Log Diff Autocorr.png'
rvp.plot_corr(standardized, 'Standardized Log Diff Autocorr', f_sl)
plot_fs.append(f_l)
plot_fs.append(f_sl)

####
# apply garch model to the time series to forecast volatility
####

# fit model, use garch model with skew-t distribution and 'shock' approach
X = t10['Diff'].dropna()
gm = arch_model(X, p=1, o=1, q=1, mean='zero', vol='GARCH', dist='skewt')
result = gm.fit(update_freq=0, disp='off')
result.summary()

# generate plot and print summary to console
f = 'Plots/model_plot.png'
rvp.plot_arch(result, f)
plot_fs.append(f)

# plot results of the model
f = 'Plots/model_results.png'
rvp.plot_model_results(result, X, f)
plot_fs.append(f)

# simulate forward returns
n_periods = 10
start_dt = '2018-12-31'
f = 'Plots/simulation.png'
rvp.plot_forecasting(result, n_periods, start_dt, f)
plot_fs.append(f)

# knit all PNGs to PDF
f = 'rate-volatility-plots.pdf'
rvf.knit_pngs_to_single_pdf(plot_fs, f)
