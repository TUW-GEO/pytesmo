# Copyright (c) 2014,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Mar 7, 2014

Plot anomalies around climatology using colors

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

import pytesmo.time_series.anomaly as anom


def plot_clim_anom(df, clim=None, axes=None, markersize=0.75,
                   mfc='0.3', mec='0.3', clim_color='0.0',
                   clim_linewidth=0.5, clim_linestyle='-',
                   pos_anom_color='#799ADA', neg_anom_color='#FD8086',
                   anom_linewidth=0.2, add_titles=True):
    """
    Takes a pandas DataFrame and calculates the climatology and anomaly
    and plots them in a nice way for each column

    Parameters
    ----------
    df : pandas.DataFrame
    clim : pandas.DataFrame, optional
        if given these climatologies will be used
        if not given then climatologies will be calculated
        this DataFrame must have the same number of columns as df
        and also the column names.
        each climatology must have doy as index.
    axes : list of matplotlib.Axes, optional
           list of axes on which each column should be plotted
           if not given a standard layout is generated
    markersize : float, optional
        size of the markers for the datapoints
    mfc : matplotlib color, optional
        markerfacecolor, color of the marker face
    mec : matplotlib color, optional
        markeredgecolor
    clim_color : matplotlib color, optional
        color of the climatology
    clim_linewidth : float, optional
        linewidth of the climatology
    clim_linestyle : string, optional
        linestyle of the climatology
    pos_anom_color : matplotlib color, optional
        color of the positive anomaly
    neg_anom_color : matplotlib color, optional
        color of the negative anomaly
    anom_linewidth : float, optional
        linewidth of the anomaly lines
    add_titles : boolean, optional
        if set each subplot will have it's column name as title
        Default : True

    Returns
    -------
    Figure : matplotlib.Figure
        if no axes were given
    axes : list of matploblib.Axes
        if no axes were given
    """

    if type(df) == pd.Series:
        df = pd.DataFrame(df)

    nr_columns = len(df.columns)

    # make own axis if necessary
    if axes is None:
        own_axis = True
        gs = gridspec.GridSpec(nr_columns, 1, right=0.8)

        fig = plt.figure(num=None, figsize=(6, 2 * nr_columns),
                         dpi=150, facecolor='w', edgecolor='k')

        last_axis = fig.add_subplot(gs[nr_columns - 1])
        axes = []
        for i, grid in enumerate(gs):
            if i < nr_columns - 1:
                ax = fig.add_subplot(grid, sharex=last_axis)
                axes.append(ax)
                ax.xaxis.set_visible(False)
        axes.append(last_axis)

    else:
        own_axis = False

    for i, column in enumerate(df):
        Ser = df[column]
        ax = axes[i]

        if clim is None:
            clima = anom.calc_climatology(Ser)
        else:
            clima = pd.Series(clim[column])
        anomaly = anom.calc_anomaly(Ser, climatology=clima, return_clim=True)

        anomaly[Ser.name] = Ser
        anomaly = anomaly.dropna()

        pos_anom = anomaly[Ser.name].values > anomaly['climatology'].values
        neg_anom = anomaly[Ser.name].values < anomaly['climatology'].values

        ax.plot(anomaly.index, anomaly[Ser.name].values, 'o',
                markersize=markersize, mfc=mfc, mec=mec)

        ax.plot(anomaly.index, anomaly['climatology'].values,
                linestyle=clim_linestyle,
                color=clim_color,
                linewidth=clim_linewidth)

        ax.fill_between(anomaly.index,
                        anomaly[Ser.name].values,
                        anomaly['climatology'].values, interpolate=True,
                        where=pos_anom, color=pos_anom_color,
                        linewidth=anom_linewidth)
        ax.fill_between(anomaly.index,
                        anomaly[Ser.name].values,
                        anomaly['climatology'].values, interpolate=True,
                        where=neg_anom, color=neg_anom_color,
                        linewidth=anom_linewidth)
        if add_titles:
            ax.set_title(column)

    if own_axis:
        return fig, axes
    else:
        return None, None
