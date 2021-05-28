#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:11:14 2021

@author: jtobin
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from arch import arch_model
import seaborn as sns
plt.style.use('seaborn-deep')
import quandl
import pickle
from statsmodels.graphics.tsaplots import plot_acf

# local location
local_dir = os.path.expanduser('~/Documents/projects/rate_volatility')
os.chdir(local_dir)

# load in authenticate quandl code
from auth_quandl import authenticate_quandl
quandl = authenticate_quandl(quandl)

# load in rates data, scrub
def load_rates_data(refresh=False):
    if refresh:
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
    
plot_latest_curve()

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
    
yc_surface()

# 10YT volatility analysis

def vol_analysis(colname):
    # subset 10Y
    df = pd.DataFrame(rates[label_vals[colname]].copy())
    df.columns = [colname]
    
    # additional data: changes in rates, decomposed daily return
    df['Diff'] = df[colname].diff()
    df['LogDiff']= np.log(df[colname]).diff().mul(100)
    df['Ret'] = df[colname].apply(lambda x: ((1 + x)**(1/360) - 1) * 100)
    df['Vol'] = df['Diff'].rolling(90).std()
    
    # plot
    fig, axes = plt.subplots(3, figsize=(10,10))
    
    # first plot--yield over time
    axes[0].plot(df[colname], label=colname + ' Yield')
    
    # second plot--changes in yield
    axes[1].plot(df['Diff'], label='Change in ' + colname + ' Yield')
    
    # third plot--rolling volatility of changes in yield
    axes[2].plot(df['Vol'], label='Rolling 30D Rate Vol')
    
    # clean up
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    [ax.legend() for ax in axes]
    
    # save
    plt.gcf().savefig(colname + '_stats.png', dpi=600)
    plt.clf()
    
    return df
    
# run vol analyses
t3m = vol_analysis('3 MO')
t2 = vol_analysis('2 YR')
t10 = vol_analysis('10 YR')
t30 = vol_analysis('30 YR')

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
hurst(t10['10 YR'].values)
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

# apply garch model to the time series to forecast volatility

# fit model
X = t10['Diff'].dropna()
gm = arch_model(X, p=1, o=1, q=1, mean='zero', vol='GARCH', dist='skewt')
result = gm.fit(update_freq=0, disp='off')
result.plot()
result.summary()

# plot distribution of residuals

# calculate standardized residuals
gm_resid = result.resid
gm_std = result.conditional_volatility
gm_resid_st = gm_resid / gm_std

# create plot
fig, ax = plt.subplots(1, figsize=(10,7))
sns.distplot(gm_resid_st, norm_hist=True, fit=stats.norm, bins=50, ax=ax)

# evaluate fit by overlaying conditional vol over daily diffs
fig, ax = plt.subplots(1)
ax.plot(X, color='gray', label='Daily Yield Change', alpha=0.4)
ax.plot(gm_std, color='darkred', label='Conditional Volatility')
ax.legend()

# plot conditional variance against rolling 30d variance
X_roll_vol = X.rolling(30).std()
fig, ax = plt.subplots(1)
ax.plot(gm_std, color='darkred', label='Conditional Volatility')
ax.plot(X_roll_vol, color='darkgreen', label='Actual Rolling 30D Volatility')
ax.legend()

# simulate forward returns

clrs = sns.color_palette('deep')
n_periods = 30

def forecasting(n_days):
    
    n_days = 30

    # forecast
    forecast = result.forecast(start=0, horizon=n_days, reindex=False,
                               method='simulation')
    paths = forecast.simulations.residual_variances[-1].T
    
    # generate plots
    fig, ax = plt.subplots(1)
    ax.plot(paths * 100, alpha=0.10, color=clrs[0])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.suptitle('Simulated 10YT Volatility Paths')
    ax.set_xlabel('Forecast Volatility Period (Days)')
    ax.set_ylabel('Forecast Volatility (%)')
    plt.gcf().savefig('simulation.png', dpi=600)
    plt.clf()
    
    # hedgehog plot
    y = np.sqrt(forecast.residual_variance.dropna())['2017-12-31':] * 100
    vol = result.conditional_volatility['2017-12-31':] * 100
    
    fig, ax = plt.subplots(1, figsize=(9,6))
    ax.plot(vol, color=clrs[0], alpha=0.5, label='Estimated Conditional Volatility')
    for i in range(0, len(vol) - n_days, n_days):
        f = y.iloc[i]
        loc = vol.index.get_loc(f.name)
        f.index = vol.index[(loc+1):(loc+n_days+1)]
        if i == 0:
            ax.plot(f, color=clrs[1], label='Forecast Volatility Path')
        else:
            ax.plot(f, color=clrs[1])
    
    # clean up
    ax.legend(loc='upper left')
    fig.suptitle('Estimated Conditional Volatility vs. 30D Forecast Volatility Paths')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    















