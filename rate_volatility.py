#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:11:14 2021

@author: jtobin
"""

import os
import git
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
import seaborn as sns
plt.style.use('seaborn-deep')
import quandl
import pickle
from statsmodels.graphics.tsaplots import plot_acf

# local location
local_dir = os.path.expanduser('~/Documents/projects/rate_volatility')
os.chdir(local_dir)

# location of remote git repository
repo_url = 'https://github.com/james-j-tobin/interest-rate-volatility.git'

# load in rates data, scrub
def load_rates_data(refresh=False):
    if refresh:
        # authenticate quandl api
        quandl.ApiConfig.api_key = 'fExJA3CeSuzM_fBVZkiJ'
        
        # load in data
        rates = quandl.get('USTREASURY/YIELD')
        
        # pickle
        with open('rates.pickle', 'wb') as f:
            pickle.dump(rates, f)
    else:
        with open('rates.pickle', 'rb') as f:
            rates = pickle.load(f)
    
    # drop 2M
    rates.drop('2 MO', axis=1, inplace=True)
    
    # convert column names to number of days. assumes 30/360
    conv = {'MO': 30, 'YR': 360}
    col_labels = rates.columns
    col_names = [int(i) * conv[j] for i, j in [x.split(' ') for x in rates.columns]]
    rates.columns = col_names
    
    return {'data': rates, 'labels': col_labels}

# load in data
data = load_rates_data(False)
rates = data['data']
labels = data['labels']
label_vals = {k: v for k, v in zip(labels, rates.columns)}

# plot latest curve

def plot_latest_curve():
    # subset latest data
    yc = rates.iloc[-1, :].T
    
    # configure plot
    fig, ax = plt.subplots(1)
    ax.plot(yc)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(yc.index)
    ax.set_xticklabels(labels)
    plt.gcf().savefig('latest_curve.png', dpi=600)
    plt.clf()
    
# plot_latest_curve()

# surface plot of curve over time

def arrange_data(df):
    # arrange into three dimensions
    X = df.columns
    Y = np.arange(len(df.index))
    Z = df.copy()
    X, Y = np.meshgrid(X, Y)
    
    return X, Y, Z

def yc_surface():
    # subset data
    rates_sub = rates[rates.index >= dt.datetime(2006, 2, 9)].copy()
    rates_sub.fillna(method='pad', inplace=True)
    X, Y, Z = arrange_data(rates_sub)
    
    # create plot
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, Z, cmap='viridis')
    
    # format axes
    ax.zaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_yticks(rates_sub.columns)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0, 4000, 500))
    ax.set_xticklabels([format(x, '%Y') for x in rates_sub.index[np.arange(0, 4000, 500)]])
    
    # adjust plot scaling
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 0.7, 1]))
    ax.view_init(elev=20, azim=-50)
    plt.gcf().savefig('yc_surface.png', dpi=600)
    plt.clf()
    
# yc_surface()

# 1Y volatility analysis

# subset 10Y
t10 = pd.DataFrame(rates[label_vals['10 YR']].copy())
t10.columns = ['10Y']

# additional data: changes in rates, decomposed daily return
t10['Diff'] = t10['10Y'].diff()
t10['LogDiff']= np.log(t10['10Y']).diff().mul(100)
t10['Ret'] = t10['10Y'].apply(lambda x: ((1 + x)**(1/360) - 1) * 100)
t10['Vol'] = t10['Diff'].rolling(60).std()

# plot
fig, axes = plt.subplots(3, figsize=(10,10))

# first plot--yield over time
axes[0].plot(t10['10Y'], label='10Y Yield')

# second plot--changes in yield
axes[1].plot(t10['Diff'], label='Change in 10Y Yield')

# third plot--rolling volatility of changes in yield
axes[2].plot(t10['Vol'], label='Rolling 30D Rate Vol')

# clean up
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
[ax.legend() for ax in axes]

# save
plt.gcf().savefig('10y_stats.png', dpi=600)
plt.clf()

# Hurst Exponent function
# credit: https://goldinlocks.github.io/ARCH_GARCH-Volatility-Forecasting/

def hurst(S):
    """
    Returns the Hurst Exponent of the series S
    """
    
    # Create the range of lag values
    lags = range(2, 100)
    
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(S[lag:], S[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0

# run hurst coefficient
hurst(t10['10Y'].values)
hurst(t10['Diff'].dropna().values)
hurst(t10['LogDiff'].dropna().values)

# hurst is close to 0, suggesting a mean reverting series. Nice!

# make plot of diffs and serial correlation
def corr_plot(X, title):
    fig, axes = plt.subplots(2, figsize=(10,8))
    fig.suptitle(title)
    
    # change in yield
    axes[0].plot(X, label='Change in Yield')
    axes[0].set_title('Change in Yield')
    
    # autocorrelation
    plot_acf(X, ax=axes[1], lags=100, zero=False)
    
    # save
    plt.gcf().savefig(title + '.png', dpi=600)
    plt.clf()

# generate autocorr plots
subset = t10['LogDiff'].dropna()
standardized = subset.sub(subset.mean()).pow(2)
corr_plot(subset, 'Log Diff Autocorr')
corr_plot(standardized, 'Standardized Log Diff Autocorr')














