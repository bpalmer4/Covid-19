# plot stuff - a series of functions to assist with (a) data ETL 
#              and (b) plotting for COVID-19 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.ticker as ticker

import datetime
import pandas as pd
import numpy as np
import math
import copy
import sys
from Henderson import Henderson
from typing import Tuple, Set, Dict, List, Union, Optional


### --- data cleaning

def hollow_mean(series: pd.Series, middle:int = None)-> float:
    """ Calculate the mean of a series, ignoring the middle 
        element. (Used for data spike detection).
        Aguments:
        - series - pandas Series - the series for which a single 
            hollow-mean will be calculated. The length of the series
            must be >= 3 and odd. 
        - middle - optional int - the integer index of the middle
            item in the series (to be excluded from the mean), 
            where series has an integer index from 0 to length-1.
        Returns: the mean as a float"""
    
    length = len(series)
    if middle is None:
        middle = int(length / 2)
    #assert(length >= 3 and length % 2)
    return (series.iloc[:middle].sum() 
            + series[middle+1:].sum()) / (length - 1)


def rolling_mean_excluding_self(series: pd.Series, window: int=15)-> pd.Series:
    """ Calculate the rolling hollow_mean() for each element in a series. 
        Note: negative items in the series will be treated as zero.
        Note: at the head of the series, items up until window/2 are
              returned as the the rolling_mean
        Note: at the tail of the series, them same mean is returned for 
              the final window/2 items
        Arguments:
        - series - pandas Series - the series for which the 
            hollow-mean will be calculated for each item.
        - window - the size of the rolling window used for calculating  
            the hollow_mean()
        Returns: a series of means"""
    
    positive = series.where(series >= 0, other = 0)
    mean = positive.rolling(window, center=True).apply(hollow_mean)
    for n in range(int(window / 2) + 1, window):
        position = -(window - n)
        mean.iloc[position] =  hollow_mean(positive[-window:], n)   
    return mean


def replace_proportional(series: pd.Series, point:pd.Timestamp, 
                         replacement: float)-> pd.Series:
    """ For a series, replace the element at .loc[point],  
        with replacement, then proportionately apply the 
        adjustment to all the elements immediately before 
        point in series, such that the sum of series is 
        unchanged. 
        Aguments:
        - series - pd.Series - the series we are going to adjust
        - point - pd.Timestamp - the point in the series we will adjust
        - replacement - float - the replacement value for the adjustment
        Assumes:
        - the index of series is unique and monotonic_increasing.
        - all elements of series before point are zero or positive.
        Returns: 
        - an ajdusted series with the same sum as the original series"""
    
    # sanity checks
    #assert(series.index.is_monotonic_increasing)
    #assert(series.index.is_unique)
    
    # replace original and remember the adjustment from the original
    adjustment = series[point] - replacement
    series.loc[point] = replacement
            
    # apply adjustment to this point and all preceeding data
    base = series.index[0]
    window = (series.index <= point)
    #assert((series[window] >= 0).all()) # a sanity check
    sum_for_window = series[window].sum()
    if sum_for_window > 0:
        factor = (sum_for_window + adjustment) / sum_for_window
        series[window] = series[window] * factor
    else:
        print(f'Warning: negative adjustment {series.name}')
        sys.exit(1)
    
    return series


def replace_nearby(series: pd.Series, point:pd.Timestamp, 
                         replacement:float=0)-> pd.Series:
    """ For a series, replace the element at .loc[point],  
        with replacement, then apply the adjustment to 
        the elements immediately before point in series,
        such that the sum of series is unchanged. 
        Aguments:
        - series - pd.Series - the series we are going to adjust
        - point - pd.Timestamp - the point in the series we will adjust
        - replacement - int - the replacement value for the adjustment
        Assumes:
        - the index of series is unique and monotonic_increasing.
        - all elements of series before point are zero or positive.
        - the element of series at point is negative
        Returns: 
        - an ajdusted series with the same sum as the original series"""

    # sanity checks
    #assert(series.index.is_monotonic_increasing)
    #assert(series.index.is_unique)
    #assert(series[point] < 0)
 
    # replace original and remember the adjustment from the original
    adjustment = series[point] - replacement
    series.loc[point] = replacement

    for p in reversed(series.index[series.index < point]):
        if adjustment >= 0:
            break
        if (local := series[p]) <= 0:
            continue
        local += adjustment
        adjustment = local
        if local < 0:
            local = 0
        series[p] = local

    # check we have fixed the adjustment
    assert(adjustment >= 0)
    
    return series


def rolling_zero_count(series: pd.Series, window: int=15)-> pd.Series:
    """ Count the number of zero or negative values nearby each 
        element in a series, where nerby is defined by the centered 
        window parameter."""
    
    return ( series
              .rolling(window, center=True)
              .apply(lambda x: (x <= 0).sum())
              .ffill() )


def negative_correct_daily(series: pd.Series, verbose)-> pd.Series:
    """ Correct negative adjustments to a series.
        Small negatives will be adjusted against immediately prior
        positives in the series. Large negatives will be adjusted 
        proportionately against the entire prior series. 
        Note: any NANs in the series will be replaced with zero (0)
        Arguments:
        - series - pd.Series - a series of new daily cases/deaths
        Returns: a corrected series"""
    
    # treat MINOR negative adjustments as local-temporal
    # adjustments, with minor defined as >= MINOR_NEG
    MINOR_NEG = -12 
    
    # zero nans and sort
    series = series.sort_index(ascending=True).fillna(0)
    
    # identify the places needing adjustment
    negatives = series[series < 0].index
    if (series[negatives] < 0).any():
        if verbose:
            print(f'There are negatives in {series.name}')
            print(f'{series[negatives]}')
        replacers = rolling_mean_excluding_self(series)
        # make the adjustments
        for fix_me in negatives:
            if series[fix_me] >= MINOR_NEG:
                series = replace_nearby(series, fix_me)
            else:
                series = replace_proportional(series, fix_me, 
                                replacement=replacers[fix_me])   
    return series


def positive_correct_daily(series: pd.Series, verbose)-> pd.Series:
    """Correct excess positive spikes in a series."""

    MIN_SPIKE = 30 # ignore smaller spikes
    SPIKE = 5 # defined as SPIKE times the local daily normal

    # consecutive data check - no point trying to deal with outliers in sparse data
    AT_LEAST_ONCE_CONSEC = 15 # n the series we have some consecutive data
    positive = (series >= 0.001).astype(int)
    id = positive.diff().ne(0).cumsum()
    max_consecutive = positive.groupby(id).cumsum().max()
    if max_consecutive < AT_LEAST_ONCE_CONSEC:
        if verbose:
            print(f'Data too sparse in {series.name} '
                  f'(max_consecutive={max_consecutive})')
        return series
    
    # identify adjustments
    #cumsum = series.cumsum()
    zeros = rolling_zero_count(series)
    rolling_mean = rolling_mean_excluding_self(series)
    spikes = (series > (SPIKE + (zeros * 2)) * rolling_mean) & (series > MIN_SPIKE) 
    spikes = series[spikes].index

    # make changes
    original = series.copy()
    if len(spikes):
        spike_frame = pd.DataFrame([series[spikes], 
                                    rolling_mean[spikes], 
                                    zeros[spikes]], 
                                    index=["spike", "mean", "zeros"])
        if verbose:
            print(f'Spikes in {series.name}\n{spike_frame}')
        
    for fix_me in spikes:
        series = replace_proportional(series, fix_me, 
                                      replacement=rolling_mean[fix_me])

    # check nothing has gone wrong
    # final check - do no harm
    ACCEPTABLE_INCREASE = 1.075 # 7.5 per cent is the max acceptable increase
    if ((series.max() > (original.max() * ACCEPTABLE_INCREASE)) 
        & (original >= 0)).all():
        # we have not made things better
        if verbose:
            print(f'Spike not fixed for {series.name} '
                  f'({series.max() / original.max()})')
        series = original

    return series


def get_corrected_daily_new(input_frame: pd.DataFrame, verbose)-> pd.DataFrame:
    output_frame = pd.DataFrame(np.nan, columns=input_frame.columns, index=input_frame.index)
    for col in input_frame.columns:
        series = negative_correct_daily(input_frame[col], verbose)
        output_frame[col] = positive_correct_daily(series, verbose)
    return output_frame


def get_uncorrected_daily_new(frame: pd.DataFrame)-> pd.DataFrame:
    
    # prepend a row of zeros - for the diff
    start = frame.index.min()
    previous = start - pd.Timedelta(days=1)
    ret_frame = frame.copy().ffill().fillna(0).T
    ret_frame[previous] = 0 # prepend a row of zeros
    ret_frame = ret_frame.T.sort_index()
    
    # diff, drop the resulting NAN row
    ret_frame = (ret_frame
                .diff()
                .dropna(how='all', axis='index')
    )
    return ret_frame


def dataframe_correction(uncorrected_cum: pd.DataFrame, 
        corrections: Optional[pd.DataFrame]=None, verbose=True)-> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Take an uncorrected dataframe of daily cumulative data,
        and return three dataframes:
        - uncorrected daily new data
        - corrected daily new data
        - corrected cumulative data"""

    nans = uncorrected_cum.isna()

    uncorrected_cum = uncorrected_cum.ffill().fillna(0)
    uncorrected_daily_new = get_uncorrected_daily_new(uncorrected_cum)
    corrected_daily_new = get_corrected_daily_new(uncorrected_daily_new, verbose)
    corrected_cumulative = corrected_daily_new.cumsum().dropna(
        how='all', axis='index')
        
    # sanity checks - cumulative totals should not have changed
    delta = 0.0001
    check = ( (uncorrected_cum.iloc[-1]-delta < corrected_cumulative.iloc[-1]) &
              (corrected_cumulative.iloc[-1] < uncorrected_cum.iloc[-1]+delta) )
    assert(check.all(), f'Check: \n{uncorrected_cum.iloc[-1]}\n'
                        f'{corrected_cumulative.iloc[-1]}')
    check2 = (((uncorrected_daily_new.sum() - delta) < corrected_daily_new.sum())
              & (corrected_daily_new.sum() < (uncorrected_daily_new.sum() + delta)))
    assert(check2.all(), f'Check: \n{uncorrected_cum.iloc[-1]}\n'
                         f'{corrected_cumulative.iloc[-1]}')
    assert(len(uncorrected_cum) == len(corrected_cumulative))
    assert(len(uncorrected_daily_new) == len(corrected_daily_new))
    
    # restore nans 
    for data in (uncorrected_daily_new, corrected_daily_new, corrected_cumulative):
        data.where(~nans, other=np.nan, inplace=True)

    return (uncorrected_daily_new,
            corrected_daily_new,
            corrected_cumulative)


### --- Plotting
    
# matplotlib stuff for date formatting xticklabels
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter
munits.registry[pd.Timestamp] = converter
munits.registry[pd.Period] = converter


def start_point(country_name):
    if country_name == 'China':
        return pd.Timestamp('2019-12-30')
    if country_name in ['South Korea', 'Iran', 'France', 'Italy', 'Spain',
                        'San Marino', 'Australia',
                        
                        'ACT', 'NT', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']:
        return pd.Timestamp('2020-02-15')
    return pd.Timestamp('2020-03-01')


def finalise_plot(ax, **kwargs):
    """A function to automate the completion of simple 
       matplotlib plots, including saving it to file 
       and closing the plot when done.
       
       Arguments:
       - ax - required - a matplotlib axes object
       - matplotlib axes settings - optional - any/all of the 
         following : 
            title, xlabel, ylabel, xticks, yticks, 
            xticklabels, yticklabels, xlim, ylim, 
            xscale, yscale, margin
       - lfooter - optional - string - left side chart footer
       - rfooter - optional - string - right side chart footer
       - tight_layout_pad - optional - float - tight layout padding
       - set_size_inches - optional - tuple of floats - plot size,
         defaults to (8, 4) if not set
       - save_as - optional - string - filename for saving
       - chart_directory - optional - string - chart directory for
         saving plot using the plot title as the name for saving.
         The save file is defined by the following string:
         f'{kwargs["chart_directory"]}{title}{save_tag}.{save_type}'
         [Note: assumes chart_directory has concluding '/' in it].
       - save_type - optional - string - defaults to 'png' if not set
       - save_tag - optional - string - additional name
       - show - whether to show the plot
       - display - whether to display the plot (from Jupyter Notebook)
       - dont_close - optional - if set and true, the plot is not 
         closed 
       
       Returns: None
    """

    # defaults
    DEFAULT_SET_SIZE_INCHES = (8, 4)
    DEFAULT_SAVE_TYPE = 'png'
    DEFAULT_TIGHT_PADDING = 1.2
    DEFAULT_SAVE_TAG = ''
    AXES_SETABLE = ('title', 'xlabel', 'ylabel', 'xticks', 'yticks', 
                    'xticklabels', 'yticklabels', 'xlim', 'ylim', 
                    'xscale', 'yscale')
    OTHER_SETABLE = ('lfooter', 'rfooter', 'tight_layout_pad', 
                     'set_size_inches', 'save_as', 'chart_directory',
                     'save_type', 'save_tag', 'show', 'display',
                     'dont_close', 'margins', 'no_locator')
    
    # utility
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)
    
    # precautionary
    if 'title' not in kwargs:
        eprint('Warning: the plot title has not been set\n'
              '\tin the call to finalise_plot().')
    for arg in kwargs:
        if arg not in AXES_SETABLE and arg not in OTHER_SETABLE:
            eprint(f'Warning: the argument "{arg}" in the call\n'
                  '\tto finalise_plot() is not recognised.')
    
    # usual axes settings
    axes_settings = {}
    for arg in kwargs:
        if arg not in AXES_SETABLE:
            continue
        axes_settings[arg] = kwargs[arg]
    if len(axes_settings):
        ax.set(**axes_settings)
    
    # margins
    if 'margins' in kwargs and kwargs['margins'] is not None:
        ax.margins(kwargs['margins'])
    
    fig = ax.figure
    
    # increase y-axis locator
    if ax.get_yaxis().get_scale() == 'linear' and 'no_locator' not in kwargs:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(11))
    
    # right footnote
    if 'rfooter' in kwargs and kwargs['rfooter'] is not None:
        fig.text(0.99, 0.005, kwargs['rfooter'],
            ha='right', va='bottom',
            fontsize=9, fontstyle='italic',
            color='#999999')
    
    # left footnote
    if 'lfooter' in kwargs and kwargs['lfooter'] is not None:
        fig.text(0.01, 0.005, kwargs['lfooter'],
            ha='left', va='bottom',
            fontsize=9, fontstyle='italic',
            color='#999999')

    # figure size
    if 'set_size_inches' in kwargs:
        size = kwargs['set_size_inches']
    else:
        size = DEFAULT_SET_SIZE_INCHES
    fig.set_size_inches(*size)
    
    # tight layout
    if 'tight_layout_pad' in kwargs:
        pad = kwargs['tight_layout_pad']
    else:
        pad = DEFAULT_TIGHT_PADDING
    fig.tight_layout(pad=pad)
    
    # save the plot to file
    # - save using the specified file name
    save_as = None
    if 'save_as' in kwargs:
        save_as = kwargs['save_as']
    
    # - save using a file name built from diretory-title-tag-type
    elif 'chart_directory' in kwargs:
        save_type = DEFAULT_SAVE_TYPE 
        if 'save_type' in kwargs:
            save_type = kwargs['save_type']
        save_tag = DEFAULT_SAVE_TAG
        if 'save_tag' in kwargs:
            save_tag = kwargs['save_tag']
        # file-system safe
        if 'title' in kwargs:
            title = kwargs['title'].replace('[:/]', '-')
        else:
            title = ''
        save_as = (f'{kwargs["chart_directory"]}{title}'
                   f'{save_tag}.{save_type}')

    # - warn if there is no saving arrangement
    else:
        eprint('Warning: in the call to finalise_plot()\n'
              '\tyou need to specify either save_as or\n'
              '\tchart_directory to save a plot to file.')

    if save_as:
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        fig.savefig(save_as, dpi=300)
    
    # show the plot
    if 'show' in kwargs and kwargs['show']:
        plt.show()

    # display the plot (from Jupyter Notebook)
    if 'display' in kwargs and kwargs['display']:
        display(fig)

    # close the plot
    if 'dont_close' not in kwargs or not kwargs['dont_close']:
        plt.close()
    
    return None


def _annotate_bars_on_chart(series, ax):
    # annotate the plot
    span = series.max() - series.min()
    inside = series.min() + span / 2.0
    spacer = span / 150.0
    for y, x in enumerate(series):
        xpos = x if x < inside else 0
        color = 'black' if xpos > 0 else 'white'
        ax.annotate(f'{x:,}', xy=(xpos+spacer, y), 
                   va='center', color=color, size='small')


def plot_barh(series, **kwargs):
    """plot series as a horizontal bar chart"""
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.barh(series.index, series, color='gray')
    _annotate_bars_on_chart(series, ax)
    ax.margins(0.01)
    finalise_plot(ax, **kwargs)


def simplify_values(series):
    s = series.copy().astype(float)
    numerical = ('', '$10^3$', '$10^6$', '$10^9$', '$10^12$', '$10^15$')
    text = ('', 'Thousand', 'Million', 'Billion', 'Trillion', 'Quadrillion')
    index = 0
    while s.max() > 1000.0:
        s = s / 1000.0
        index += 1
    return s, numerical[index], text[index]

    
def plot_orig_smooth(orig, n, mode, name, **kwargs):
    """plot series in original and smoothed forms,
       return the smoothed series
       Note: kwargs['ylabel'] is overwritten"""
    
    # get series into simplified values and create ylabel
    orig, _, text = simplify_values(orig)
    label = f'{text} daily new {mode}'.strip().capitalize()
    kwargs['ylabel'] = label
    
    # Henderson moving average
    hendo = Henderson(orig.dropna(), n)
    start = start_point(name)
    hendo = hendo[hendo.index >= start]   
    ax = hendo.plot.line(lw=2, color="hotpink", 
                          label=f'{n}-term Henderson moving average')

    # simple rolling average
    m = 7
    smooth = orig.rolling(m, center=True).mean()
    smooth = smooth[smooth.index >= start]   
    smooth.plot.line(lw=2, color="darkorange", ax=ax,
                          label=f'{m}-term rolling average')

    # original data
    orig = orig[orig.index >= start]   
    orig.plot.line(lw=1, color='royalblue', ax=ax, 
                  label=label)
                  
    # final touches
    ax.legend(loc='best')
    xlim = ax.get_xlim()
    adj = (xlim[1] - xlim[0]) * 0.01
    ax.set_xlim(xlim[0]-adj, xlim[1]+adj)
    finalise_plot(ax, **kwargs)
    return hendo


# ---    

def scale_zero_infinity(s: pd.Series) -> pd.Series:
    """ Scales numbers between zero and infinity to numbers
        between zero and two, symmetrically around one."""
    assert( (s.isna() | s>=0).all() )
    return s.where(s<=1, other=2-(1.0/s))


def plot_growth_factor(new_: pd.Series, period=5, **kwargs):
    """Plot period on period growth in new_ using a non-linear growth factor axis
       Uses the same **kwargs as finalise_plot()
       Note: if rfooter in **kwargs, it will be over-written.
       Returns the calculated (and unscaled) growth factor."""

    MARGINS = 0.02
    mode = 'volume'
    if 'mode' in kwargs:
        mode = kwargs['mode']
        del kwargs['mode']
        
    # a useful warning
    if new_.isna().sum():
        print('Warning: unexpected NaNs in input series in plot_growth_factor()')
    
    # we use volume as a plot background 
    volume = new_.rolling(period, min_periods=1).sum(skipna=True)
    gf_original = volume / volume.shift(period)

    # remove the first period of infinite (undefined) growth
    gf_original = gf_original.replace(np.inf, np.nan)

    # trim series to a common start date
    start = start_point(new_.name)
    gf_trimmed = gf_original[gf_original.index >= start]
    volume = volume[volume.index >= start]
    
    # adjusted scale to be symetric around 1
    gf_scaled = scale_zero_infinity(gf_trimmed)
    
    # recent growth charts 
    if 'recent' in kwargs:
        recent = kwargs['recent']
        gf_scaled = gf_scaled.iloc[-recent:]
        volume = volume.iloc[-recent:]
        del kwargs['recent']

    # let's not produce empty charts
    if gf_scaled.isna().all():
        return None
        
    # calibrate volume
    volume, _, volume_text = simplify_values(volume)

    # --- plot above and below 1 in different colours 
    # - resample to hourly to do this
    # - this code is a bit of a hack
    gf_forward = gf_scaled.resample('H').interpolate(method='linear', 
                                                     limit=23,
                                                     limit_area="inside",
                                                     limit_direction='forward')
    gf_back = gf_scaled.resample('H').interpolate(method='linear', 
                                                  limit=23, 
                                                  limit_area="inside",
                                                  limit_direction='backward')
    gf_hourly = gf_forward.where(gf_forward.notna() & gf_back.notna(), 
                                 other=np.nan) 
    below_hourly = gf_hourly.where(gf_hourly <= 1, other=np.nan)                 
    
    # plot
    fig, ax_left = plt.subplots()

    # plot growth on the left hand side
    ax_left.axhline(1, lw=1, color='gray', zorder=20)
    ax_left.plot(gf_hourly.index, gf_hourly, lw=2, 
                 color='#dd0000', label='Growth (>1)', zorder=30)
    ax_left.plot(below_hourly.index, below_hourly, lw=2, 
                 color='seagreen', label='Decline (<1)', zorder=40)

    # plot volume on the right hand side
    volume_label = f"{volume_text} {mode} ({period}-day rolling sum)".strip().capitalize()
    ax_right = ax_left.twinx()
    ax_right.stackplot(volume.index, volume, color='#cccccc', 
                  labels=[volume_label], zorder=10)
    ax_right.set_ylabel(volume_label)
    ax_right.grid(False)

    # order right at bottom, left on top of right
    ax_right.set_zorder(0) # bottom
    ax_left.set_zorder(1) # top
    ax_left.patch.set_visible(False)
    ax_right.patch.set_visible(True)
    
    # adjust y-axes to be of the same physical scale
    span_right = volume.max() - volume.min()
    adj_right = MARGINS * span_right
    ax_right.set_ylim(0 - adj_right, volume.max() + adj_right)
    span_left = 2
    adj_left = MARGINS * span_left
    ax_left.set_ylim(0 - adj_left, 2 + adj_left)

    # adjust x-axis
    ax_left.set_xlim(gf_scaled.index.min(), gf_scaled.index.max())
    xlim = ax_left.get_xlim()
    adj = (xlim[1] - xlim[0]) * MARGINS
    ax_left.set_xlim(xlim[0]-adj, xlim[1]+adj)
    
    # non-linear left ticks
    ax_left.set_yticks([0, 1/6, 1/3, 1/2, 3/4, 1, 2-3/4, 2-1/2, 
                        2-1/3, 2-1/6, 2])
    ax_left.set_yticklabels(["0", "$\\frac{1}{6}$", "$\\frac{1}{3}$", 
                             "$\\frac{1}{2}$", "$\\frac{3}{4}$", "1", 
                             "$\\frac{4}{3}$", "2", "3", "6", "$\infty$"])

    # remove xlabel if not present
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = None
    
    # annotate non-linear growth scale
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Growth factor'
    kwargs['ylabel'] += '\n(non-linear scale)'

    # combined legend
    loc = 'upper left'
    if 'loc' in kwargs:
        loc = kwargs['loc']
        del kwargs['loc']
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    legend = ax_left.legend(h1+h2, l1+l2, loc=loc, 
                            fontsize='small')
    legend.set_zorder(100) # at the very top on the left 
    frame = legend.get_frame()
    frame.set_facecolor('white')

    # add latest growth to right footer
    if 'rfooter' not in kwargs:
        kwargs['rfooter'] = ''
    kwargs['rfooter'] += (' ($GF_{end}=' 
                          f'{np.round(gf_original[-1], 2)}$)')
    
    # add explainer to left footer
    if 'lfooter' not in kwargs:
        kwargs['lfooter'] = f'GF = total {mode} this period / total for prev. period'

    finalise_plot(ax_left, **kwargs)
    return gf_original 

# ---
    
def plot_new_cum(new: pd.Series, cum:pd.Series, 
                 mode: str, name: str, period: str, 
                 dfrom="2020-01-21", **kwargs, ):
    
    # adjust input data for large numbers
    new, numerical, text = simplify_values(new)
    cum, cnumerical, ctext = simplify_values(cum)
    new_legend_label = f'{text} new {mode}/{period} (left)'.strip().capitalize()
    cum_legend_label = f'{ctext} cumulative {mode} (right)'.strip().capitalize()
    new_ylabel = f'{numerical} new {mode.lower()}/{period}'.strip().capitalize()
    cum_ylabel = f'{cnumerical} cumulative {mode}'.strip().capitalize()
    
    # adjust START
    DISPLAY_FROM = pd.Timestamp(dfrom)
    new = new[new.index >= DISPLAY_FROM]
    cum = cum[cum.index >= DISPLAY_FROM]
    
    # plot new
    widths = {
        'day': 0.8,
        'week': 5,
    }
    fig, ax = plt.subplots()
    ax.margins(x=0.015)
    ax.bar(new.index, new.values, width=widths[period],
           color='#dd0000', label=new_legend_label)
    ax.set_xlabel(None)

    # plot cumulative
    axr = ax.twinx()
    axr.plot(cum.index, cum.values,
             lw=2.0, color='#0000dd', ls='--',
             label=cum_legend_label)
    axr.set_ylabel(None)
    axr.grid(False)

    # add a legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='best', fontsize='small')

    # adjust y-limits to be prettier, 
    # this adjustment should not be needed, but it is
    if new.min() < 0:
        print(f'Warning: Minimum new value less than zero')
    if cum.min() < 0:
        print(f'Warning: Minimum cum value less than zero')
    MARGIN = 1.025
    ylim = 0, (new.max() * MARGIN)
    ax.set_ylim(ylim)
    ylimr = 0, (cum.max() * MARGIN)
    axr.set_ylim(ylimr)
    
    # This makes the dates for xticklabels look a little nicer
    locator = mdates.AutoDateLocator(minticks=4, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # y-axis labels - the hard way
    fig = ax.figure
    lHeight = 0.96
    lInstep = 0.02
    fig.text(1.0-lInstep, lHeight, cum_ylabel,
            ha='right', va='top', fontsize=11,
            color='#333333')
    fig.text(lInstep, lHeight, new_ylabel,
            ha='left', va='top', fontsize=11,
            color='#333333')

    finalise_plot(ax, **kwargs)

    
# ---
    
def _get_day(series):
    # find the last day of a pandas Series or DataFrame
    last_day = series.dropna().index[-1] # last day with data
    return {
        0: 'W-MON',
        1: 'W-TUE',
        2: 'W-WED',
        3: 'W-THU',
        4: 'W-FRI',
        5: 'W-SAT',
        6: 'W-SUN',
    }[last_day.dayofweek]


def _get_weekly(daily_series):
    
    rule = _get_day(daily_series)

    # convert the data to weekly
    weekly = daily_series.dropna().resample(rule=rule, closed='right').sum()
    cum_weekly = weekly.cumsum(skipna=True)

    # reindex by half a week to centre labels on bars
    weekly.index = weekly.index - pd.Timedelta(3.5, unit='d')
    cum_weekly.index = cum_weekly.index - pd.Timedelta(3.5, unit='d')
    
    return rule, weekly, cum_weekly


def plot_weekly(daily, mode, data_quality, dfrom="2020-01-21", **kwargs):
    """Plot weekly bar charts for daily new cases and deaths
        Function paramters:
        - daily is a DataFrame of daily timeseries data
        - mode is one of 'cases' or 'deaths' 
        - data_quality is a Series of strings,
            used for the left footnote on charts
        - dfrom is a date string to display from
        Returns: None """
    
    for name, series in daily.iteritems():
    
        if series.sum(skipna=True) == 0: 
            continue # avoid empty plots
        
        rule, weekly, weekly_cum = _get_weekly(series)
        plot_new_cum(
            weekly, 
            weekly_cum, 
            mode, 
            name,
            'week',
            dfrom, 
            title=f'COVID-19 {name} {mode.title()}',
            rfooter=data_quality[name],
            lfooter=f'Total {mode.lower()}: '
                    f'{int(weekly_cum[-1]):,}; '
                    f'(WE={rule[-3:].title()})',
            **kwargs,
        )
        
    return None


# ---
   
def plot_regional_per_captia(new_df, mode, regions, population, **kwargs):
    # constants
    ROLLING_AVE = 7 # days
    POWER = 6
    PER_CAPITA = 10 ** POWER # population

    k_copy = copy.deepcopy(kwargs)

    # data titdy-up
    df = new_df.rolling(ROLLING_AVE, center=True).mean() # smooth
    df = df.div(population / PER_CAPITA, axis=1) # per capita

    # rework
    if not 'ylabel' in k_copy:
        k_copy['ylabel'] = (f'Daily new {mode} per $10^{POWER}$ population'+
                          f'\nCentred {ROLLING_AVE}-day rolling average')
    if not 'xlabel' in k_copy:
        k_copy['xlabel'] = None
            
    saved_k = copy.deepcopy(k_copy)
    for region, states in regions.items():
  
        # plot context in light grey
        ax = df.plot(c='#aaaaaa', lw=0.5)
        ax.get_legend().remove()
       
        k_copy['title'] = f'COVID-19 Daily New {mode.title()} - {region}'
        
        # plot this group in color
        subset = df[states]
        ax_new = ax.twinx()
        subset.plot(ax=ax_new, linewidth=2.5, legend=True)
        ax_new.legend(title=None, loc="upper left")
        ax_new.grid(False)
        ax_new.set_yticklabels([])
        ax_new.set_ylim(ax.get_ylim())
         
        # finalise
        finalise_plot(ax, **k_copy)

        # next loop
        k_copy = copy.deepcopy(saved_k)
