#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:30:30 2021

@author: jtobin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import rate_volatility_functions as rvf
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from scipy import stats
plt.style.use('seaborn-deep')
clrs = sns.color_palette('deep')


def plot_latest_curve(rates_data, data_labels, f):
    """
    Plots latest yield curve present in rates_data dataframe.
    """

    # subset latest data
    yc = rates_data.iloc[-1, :].T

    # configure plot
    fig, ax = plt.subplots(1)
    ax.plot(yc)
    ax.yaxis.set_major_formatter(tck.PercentFormatter())
    ax.set_xticks(yc.index)
    ax.set_xticklabels(data_labels)
    fig.suptitle('Latest Yield Curve')
    ax.set_ylabel('Annualized Nominal Yield')

    # save, close
    plt.gcf().savefig(f, dpi=600)
    plt.close()

    return


def plot_yc_surface(rates_data, start_dt, data_labels, f):
    """
    Creates a 3D surface plot of the yield curve over time. Yield on the
    'height' axis, tenor on the 'width' axis, and date on the 'depth' axis.
    """

    # subset data
    rates_sub = rates_data[rates_data.index >= start_dt].copy()
    rates_sub.fillna(method='pad', inplace=True)
    X, Y, Z = rvf.arrange_data(rates_sub)

    # create plot
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, Z, cmap='viridis')

    # format axes
    ax.zaxis.set_major_formatter(tck.PercentFormatter())
    ax.set_yticks(rates_sub.columns)
    ax.set_yticklabels(data_labels)
    ax.set_xticks(np.arange(0, 4000, 500))
    ax.set_xticklabels([format(x, '%Y') for x in rates_sub.index[np.arange(0, 4000, 500)]])
    fig.suptitle('Surface Plot of Yield Curve Over Time')

    # adjust plot scaling
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 0.7, 1]))
    ax.view_init(elev=20, azim=-50)

    # save, close
    plt.gcf().savefig(f, dpi=600)
    plt.close()

    return


def plot_yld_vol(rates_data, label_values, colname, f):
    """
    Creates a multi-plot of the yield over time on plot 1 and the rolling 90
    day rate volatility on plot 2.
    """

    # subset 10Y
    df = pd.DataFrame(rates_data[label_values[colname]].copy())
    df.columns = [colname]

    # additional data: changes in rates, decomposed daily return
    df['Diff'] = df[colname].diff()
    df['LogDiff'] = np.log(df[colname]).diff() * 100
    df['Ret'] = df[colname].apply(lambda x: ((1 + x)**(1/360) - 1) * 100)
    df['Vol'] = df['Diff'].rolling(90).std()

    # plot setup
    fig, axes = plt.subplots(3, figsize=(10, 10))
    fig.suptitle(colname + ' Treasury Yield')

    # first plot--yield over time
    axes[0].plot(df[colname], label=colname + ' Yield')
    axes[0].set_title('Nominal Yield')

    # second plot--changes in yield
    axes[1].plot(df['Diff'], label='Change in ' + colname + ' Yield')
    axes[1].set_title('Daily Change in Nominal Yield')

    # third plot--rolling volatility of changes in yield
    axes[2].plot(df['Vol'], label='Rolling 90D Rate Vol')
    axes[2].set_title('Rolling 90D Yield Volatility')

    # clean up
    [ax.yaxis.set_major_formatter(tck.PercentFormatter()) for ax in axes]
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.08, right=0.92)
    [ax.legend() for ax in axes]

    # save, close
    plt.gcf().savefig(f, dpi=600)
    plt.close()

    return df


def plot_corr(X, title, f):
    """
    Creates a multi-plot with the daily change in yield on plot 1 and the
    serial correlation of the daily changes on plot 2.
    """

    # create subplots arrangement
    fig, axes = plt.subplots(2, figsize=(10, 8))
    fig.suptitle(title)

    # change in yield
    axes[0].plot(X, label='Change in Yield')
    axes[0].set_title('Change in Yield')

    # autocorrelation
    plot_acf(X, ax=axes[1], lags=100, zero=False)

    # save
    plt.gcf().savefig(f, dpi=600)
    plt.close()

    return


def plot_arch(model_result, f):
    """
    Plots the result of a fitted ARCH model. Plots standardized residuals
    on plot 1 and conditional volatility on plot 2.
    """

    # plot model, save
    model_result.plot()
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.savefig(f, dpi=600)
    plt.close()

    return


def plot_model_results(model_result, data, f):
    """
    Creates a multi-plot of more results from the fitted ARCH model. Plots
    histogram of standardized residuals on plot 1, an overlay of conditional
    volatility over actual daily yield changes on plot 2, and actual rolling
    30-day volatility on plot 3.
    """

    # calculate standardized residuals
    gm_resid = model_result.resid
    gm_std = model_result.conditional_volatility
    gm_resid_st = gm_resid / gm_std

    # create plot
    fig, axes = plt.subplots(3, figsize=(10, 12))
    fig.suptitle('10Y Yield Volatility ARCH Model Results')
    sns.distplot(gm_resid_st, norm_hist=True, fit=stats.norm,
                 bins=50, ax=axes[0])
    axes[0].set_title('Distribution of Daily Yield Changes')

    # evaluate fit by overlaying conditional vol over daily diffs
    axes[1].plot(data, color='gray', label='Daily Yield Change', alpha=0.4)
    axes[1].plot(gm_std, color='darkred', label='Conditional Vol.')
    axes[1].set_title('Conditional Vol Over Daily Yield Change')

    # plot conditional variance against rolling 30d variance
    X_roll_vol = data.rolling(30).std()
    axes[2].plot(gm_std, color=clrs[0], label='Conditional Vol.')
    axes[2].plot(X_roll_vol, color=clrs[1], label='Actual Rolling 30D Vol.')
    axes[2].set_title('Rolling Volatility Over ARCH Conditional Volatility')

    # clean up
    [ax.legend() for ax in axes]

    # save
    plt.gcf().savefig(f, dpi=600)
    plt.close()

    return


def plot_forecasting(model_result, n_days, start_date, fn):
    """
    Uses the fitted ARCH model to forecast n_days future values. Start date
    is the day from which to display the results of the forecast on a plot.
    The plot includes forecast volatility on plot 1 and a hedgehog plot of
    conditional volatility and forecast volatility on plot 2. 
    """
    
    # forecast
    forecast = model_result.forecast(start=0, horizon=n_days, reindex=False,
                                     method='simulation')
    paths = forecast.simulations.residual_variances[-1].T

    # generate plots
    fig, axes = plt.subplots(2, figsize=(10, 8))

    # plot forecast volatility
    axes[0].plot(paths * 100, alpha=0.10, color=clrs[0])
    axes[0].yaxis.set_major_formatter(tck.PercentFormatter())
    axes[0].set_xlabel('Forecast Volatility Period (Days)')
    axes[0].set_ylabel('Forecast Volatility (%)')
    axes[0].set_title('Simulated 10YT Volatility Paths')

    # make hedgehog plot

    # residual variance and conditional volatility
    y = np.sqrt(forecast.residual_variance.dropna())[start_date:] * 100
    vol = model_result.conditional_volatility[start_date:] * 100

    # add to plot; only add label for first line
    axes[1].plot(vol, color=clrs[0], alpha=0.5, label='Est. Conditional Vol.')
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
    axes[1].yaxis.set_major_formatter(tck.PercentFormatter())

    # save
    plt.gcf().savefig(fn, dpi=600)
    plt.close()

    return
