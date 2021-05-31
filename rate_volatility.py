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

# make directory for plots
if not os.path.exists('Plots'):
    os.mkdir('Plots')
    
# empty list for plot paths for later combination into a PDF
plot_fs = []

# load in rates data, scrub ---------------------------------------------------

def load_rates_data(refresh=False):
    if refresh:
        # load in data
        rates = quandl.get('USTREASURY/YIELD')
        
        # pickle
        with open('rates.pickle', 'wb') as f:
            pickle.dump(rates, f)
    else:
        # open from pickle
        with open('rates.pickle', 'rb') as f:
            rates = pickle.load(f)
    
    # drop 2 month yield
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

# plot latest curve -----------------------------------------------------------

def plot_latest_curve(rates_data):
    # subset latest data
    yc = rates_data.iloc[-1, :].T
    
    # configure plot
    fig, ax = plt.subplots(1)
    ax.plot(yc)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(yc.index)
    ax.set_xticklabels(labels)
    
    # save, close
    f = 'Plots/latest_curve.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return f
    
f = plot_latest_curve(rates)
plot_fs.append(f)

# surface plot of curve over time ---------------------------------------------

def arrange_data(df):
    # arrange into three dimensions
    X = df.columns
    Y = np.arange(len(df.index))
    Z = df.copy()
    X, Y = np.meshgrid(X, Y)
    
    return X, Y, Z

def yc_surface(rates_data, start_dt, data_labels):
    # subset data
    rates_sub = rates_data[rates_data.index >= start_dt].copy()
    rates_sub.fillna(method='pad', inplace=True)
    X, Y, Z = arrange_data(rates_sub)
    
    # create plot
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, Z, cmap='viridis')
    
    # format axes
    ax.zaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_yticks(rates_sub.columns)
    ax.set_yticklabels(data_labels)
    ax.set_xticks(np.arange(0, 4000, 500))
    ax.set_xticklabels([format(x, '%Y') for x in rates_sub.index[np.arange(0, 4000, 500)]])
    
    # adjust plot scaling
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 0.7, 1]))
    ax.view_init(elev=20, azim=-50)
    
    # save, close
    f = 'Plots/yc_surface.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return f

# run yield curve surface plot
yc_date = dt.datetime(2006, 2, 9)
f = yc_surface(rates, yc_date, labels)
plot_fs.append(f)

# 10YT volatility plots -------------------------------------------------------

def plot_yld_vol(rates_data, label_values, colname):
    # subset 10Y
    df = pd.DataFrame(rates_data[label_values[colname]].copy())
    df.columns = [colname]
    
    # additional data: changes in rates, decomposed daily return
    df['Diff'] = df[colname].diff()
    df['LogDiff'] = np.log(df[colname]).diff() * 100
    df['Ret'] = df[colname].apply(lambda x: ((1 + x)**(1/360) - 1) * 100)
    df['Vol'] = df['Diff'].rolling(90).std()
    
    # plot setup
    fig, axes = plt.subplots(3, figsize=(10,10))

    # first plot--yield over time
    axes[0].plot(df[colname], label=colname + ' Yield')
    
    # second plot--changes in yield
    axes[1].plot(df['Diff'], label='Change in ' + colname + ' Yield')
    
    # third plot--rolling volatility of changes in yield
    axes[2].plot(df['Vol'], label='Rolling 30D Rate Vol')
    
    # clean up
    [ax.yaxis.set_major_formatter(mtick.PercentFormatter()) for ax in axes]
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    [ax.legend() for ax in axes]
    
    # save, close
    f = 'Plots/' + colname + '_stats.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return df, f
    
# run vol analyses
t3m, t3m_f = plot_yld_vol(rates, label_vals, '3 MO')
t2, t2_f = plot_yld_vol(rates, label_vals, '2 YR')
t10, t10_f = plot_yld_vol(rates, label_vals, '10 YR')
t30, t30_f = plot_yld_vol(rates, label_vals, '30 YR')
plot_fs.append(t3m_f)
plot_fs.append(t2_f)
plot_fs.append(t10_f)
plot_fs.append(t30_f)

# Hurst Exponent function -----------------------------------------------------

def hurst(X):
    """
    Returns the Hurst Exponent of the series X
    credit: https://goldinlocks.github.io/ARCH_GARCH-Volatility-Forecasting/
    """
    
    # Create the range of lag values
    lags = range(2, 100)
    
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(X[lag:], X[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2

# run hurst coefficient
hurst(t10['10 YR'].values)
hurst(t10['Diff'].dropna().values)
hurst(t10['LogDiff'].dropna().values)

# ->>> hurst is close to 0, suggesting a mean reverting series. Nice!

# make plot of diffs and serial correlation -----------------------------------

def corr_plot(X, title):
    # create subplots arrangement
    fig, axes = plt.subplots(2, figsize=(10,8))
    fig.suptitle(title)
    
    # change in yield
    axes[0].plot(X, label='Change in Yield')
    axes[0].set_title('Change in Yield')
    
    # autocorrelation
    plot_acf(X, ax=axes[1], lags=100, zero=False)
    
    # save
    f = 'Plots/' + title + '.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return f

# create standardized version of log diffs
subset = t10['LogDiff'].dropna()
standardized = subset.sub(subset.mean()) ** 2

# make plots
f_l = corr_plot(subset, 'Log Diff Autocorr')
f_sl = corr_plot(standardized, 'Standardized Log Diff Autocorr')
plot_fs.append(f_l)
plot_fs.append(f_sl)

# apply garch model to the time series to forecast volatility -----------------

# fit model, use garch model with skew-t distribution and 'shock' approach
X = t10['Diff'].dropna()
gm = arch_model(X, p=1, o=1, q=1, mean='zero', vol='GARCH', dist='skewt')
result = gm.fit(update_freq=0, disp='off')
result.summary()

# generate plot and print summary to console

def arch_plot(model_result):
    # plot model, save
    model_result.plot()
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    f = 'Plots/model_plot.png'
    fig.savefig(f, dpi=600)
    plt.close()
    
    return f

f = arch_plot(result)
plot_fs.append(f)

# plot distribution of residuals ----------------------------------------------

def plot_model_results(model_result, data):
    # calculate standardized residuals
    gm_resid = model_result.resid
    gm_std = model_result.conditional_volatility
    gm_resid_st = gm_resid / gm_std
    
    # create plot
    fig, axes = plt.subplots(3, figsize=(10,12))
    clrs = sns.color_palette('deep')
    sns.distplot(gm_resid_st, norm_hist=True, fit=stats.norm, bins=50, ax=axes[0])
    
    # evaluate fit by overlaying conditional vol over daily diffs
    axes[1].plot(data, color='gray', label='Daily Yield Change', alpha=0.4)
    axes[1].plot(gm_std, color='darkred', label='Conditional Volatility')
    
    # plot conditional variance against rolling 30d variance
    X_roll_vol = data.rolling(30).std()
    axes[2].plot(gm_std, color=clrs[0], label='Conditional Volatility')
    axes[2].plot(X_roll_vol, color=clrs[1], label='Actual Rolling 30D Volatility')
    
    # clean up
    [ax.legend() for ax in axes]
    
    # save
    f = 'Plots/model_results.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return f
    
plot_model_results(result, X)
plot_fs.append(f)

# simulate forward returns ----------------------------------------------------

def forecasting(model_result, n_days, start_date):
    # forecast
    forecast = model_result.forecast(start=0, horizon=n_days, reindex=False, method='simulation')
    paths = forecast.simulations.residual_variances[-1].T
    
    # generate plots
    clrs = sns.color_palette('deep')
    fig, axes = plt.subplots(2)
    
    # plot forecast volatility
    axes[0].plot(paths * 100, alpha=0.10, color=clrs[0])
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_xlabel('Forecast Volatility Period (Days)')
    axes[0].set_ylabel('Forecast Volatility (%)')
    axes[0].set_title('Simulated 10YT Volatility Paths')

    # make hedgehog plot
    
    # residual variance and conditional volatility
    y = np.sqrt(forecast.residual_variance.dropna())[start_date:] * 100
    vol = model_result.conditional_volatility[start_date:] * 100
    
    # add to plot; only add label for first line
    axes[1].plot(vol, color=clrs[0], alpha=0.5, label='Est. Conditional Volatility')
    for i in range(0, len(vol) - n_days, n_days):
        f = y.iloc[i]
        loc = vol.index.get_loc(f.name)
        f.index = vol.index[(loc+1):(loc+n_days+1)]
        if i == 0:
            axes[1].plot(f, color=clrs[1], label='Forecast Volatility Path')
        else:
            axes[1].plot(f, color=clrs[1])
    
    # clean up
    [ax.legend(loc='upper left') for ax in axes]
    axes[1].set_title('Estimated Conditional Volatility vs. 30D Forecast Volatility Paths')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # save
    f = 'Plots/simulation.png'
    plt.gcf().savefig(f, dpi=600)
    plt.close()
    
    return f
    
n_periods = 10
start_dt = '2018-12-31'
f = forecasting(result, n_periods, start_dt)
plot_fs.append(f)
