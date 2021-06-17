#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:23:30 2021

@author: jtobin
"""

import quandl
import numpy as np
import pickle
from PIL import Image


def load_rates_data(refresh=False):
    """
    Loads in the latest treasury yield data from Quandl. Performs basic data
    cleaning functions and converts period labels to number of days.
    """

    # load in data depending on whether set to do so.
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


def arrange_data(df):
    """
    Arranges a panel dataset (cross sectional time series) into three objects
    that can be used to generate a 3D surface plot, each representing one axis
    on an X-Y-Z three-axis plot.
    """

    # arrange into three dimensions
    X = df.columns
    Y = np.arange(len(df.index))
    Z = df.copy()
    X, Y = np.meshgrid(X, Y)

    return X, Y, Z


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


def knit_pngs_to_single_pdf(pngs, f):
    """
    Knits the resulting PNG files into a single PDF file
    """

    # convert pngs to pdfs
    pdfs = {}
    i = 1
    for png in pngs:
        print('Converting ' + png)
        pdfs['img' + str(i)] = Image.open(png)
        pdfs['img' + str(i)] = pdfs['img' + str(i)].convert('RGB')
        i += 1

    # compile PDFs together
    pdfs['img1'].save(f, save_all=True, append_images=list(pdfs.values())[1:])

    return
