# plot stuff
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import datetime
import pandas as pd
import numpy as np
import math
import copy
from Henderson import Henderson
from typing import Tuple, Set, Dict, List, Union, Optional

SHOW=False

### --- data cleaning

#def restore_leading_nans(frame: pd.DataFrame)-> pd.DataFrame:
#    for col in frame.columns:
#        series = frame[col]
#        if series[0] != 0:
#            continue # there are no leading zeros
#        # reinstate leading nan values
#        if series.ne(0).sum() == 0:
#           continue # to be sure
#        index = series[series.ne(0)].index[0]
#        position = frame.index.get_loc(index)
#        frame[col][0:position] = np.nan
#    return frame


def hollow_mean(series: pd.Series, middle:int = None)-> pd.Series:
    """ Calculate the mean of a series, ignoring the middle element. """
    length = len(series)
    if middle is None:
        middle = int(length / 2)
    assert(length >= 3 and length % 2)
    return (series.iloc[:middle].sum() + series[middle+1:].sum()) / (length - 1)


def rolling_mean_excluding_self(series: pd.Series, WINDOW: int=15)-> pd.Series:
    """ Calculate the hollow_mean() for each element in a series. """
    
    # work with a positive series so negativ spikes don't set us off
    positive = series.where(series >= 0, other = 0)
    mean = positive.rolling(WINDOW, center=True).apply(hollow_mean)
    for n in range(int(WINDOW / 2) + 1, WINDOW):
        position = -(WINDOW - n)
        mean.iloc[position] =  hollow_mean(positive[-WINDOW:], n)   
    return mean


def replace_proportional(series, point, replacement)-> pd.Series:
    # replace original and remember the adjustment from the original
    adjustment = series[point] - replacement
    series.loc[point] = replacement
            
    # apply adjustment to this point and all preceeding data
    base = series.index[0]
    window = ((series.index >= base) & (series.index <= point))
    sum_for_window = series[window].sum()
    if sum_for_window > 0:
        factor = (sum_for_window + adjustment) / sum_for_window
        series[window] = series[window] * factor
    else:
        print(f'Warning: negative adjustment {series.name}')
        assert(False)
        series[window] = 0
    
    return series


def replace_nearby(series, point, replacement)-> pd.Series:
    # replace original and remember the adjustment from the original
    adjustment = series[point] - replacement
    assert(adjustment < 0) # should only be for taking away adjustments
    series.loc[point] = replacement
    base = series.index[0]

    while adjustment <= 0 and point > base:
        point = point - pd.Timedelta(days=1)
        if (local := series[point]) <= 0:
            continue
        local += adjustment
        adjustment = local
        if local < 0:
            local = 0
        series[point] = local
        
    return series


def rolling_zero_count(series: pd.Series, WINDOW: int=15)-> pd.Series:
    return series.rolling(WINDOW, center=True).apply(lambda x: (x <= 0).sum()).ffill()


def negative_correct_daily(series: pd.Series)-> pd.Series:
    """Correct negative adjustments to a series."""
    
    # Thresholds ... for correction
    MINOR_NEG = -10 # treat MINOR negative adjustments as local-temporal
    
    # zero nans and sort
    series = series.sort_index(ascending=True).fillna(0)
    rolling_mean = rolling_mean_excluding_self(series)
    
    # identify adjustments
    negatives = series[series < 0].index
    
    # make changed
    original = series.copy()
    if len(negatives):
        print(f'Negatives in {series.name}\n{series[negatives]}')
    for fix_me in negatives:
        if series[fix_me] < 0 and series[fix_me] >= MINOR_NEG:
            series = replace_nearby(series, fix_me, replacement=0)
        else:
            series = replace_proportional(series, fix_me, replacement=rolling_mean[fix_me])

    return series

def positive_correct_daily(series: pd.Series)-> pd.Series:
    """Correct excess positive spikes in a series."""

    MIN_SPIKE = 30 # ignore smaller spikes
    SPIKE = 5 # defined as SPIKE times the local daily normal

    # consecutive data check - no point trying to deal with outliers in sparse data
    AT_LEAST_ONCE_CONSEC = 15 # somewhere in the series we have some consecutive data
    positive = (series >= 0.001).astype(int)
    id = positive.diff().ne(0).cumsum()
    max_consecutive = positive.groupby(id).cumsum().max()
    if max_consecutive < AT_LEAST_ONCE_CONSEC:
        print(f'Data too sparse in {series.name} (max_consecutive={max_consecutive})')
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
        spike_frame = pd.DataFrame([series[spikes], rolling_mean[spikes], zeros[spikes]], 
                                  index=["spike", "mean", "zeros"])
        print(f'Spikes in {series.name}\n{spike_frame}')
    for fix_me in spikes:
        series = replace_proportional(series, fix_me, replacement=rolling_mean[fix_me])

    # check nothing has gone wrong
    # final check - do no harm
    ACCEPTABLE_INCREASE = 1.075 # 7.5 per cent is the Max acceptable increase
    if (series.max() > (original.max() * ACCEPTABLE_INCREASE)) & (original >= 0).all():
        # we have not made things better
        print(f'Spike not fixed for {series.name} ({series.max() / original.max()})')
        series = original

    return series


def get_corrected_daily_new(input_frame: pd.DataFrame)-> pd.DataFrame:
    output_frame = pd.DataFrame()
    for col in input_frame.columns:
         series = negative_correct_daily(input_frame[col])
         output_frame[col] = positive_correct_daily(series)
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
        corrections: Optional[pd.DataFrame]=None)-> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Take an uncorrected dataframe of daily cumulative data,
        and return three dataframes:
        - uncorrected daily new data
        - corrected daily new data
        - corrected cumulative data"""

    uncorrected_cum = uncorrected_cum.ffill().fillna(0) 
    uncorrected_daily_new = get_uncorrected_daily_new(uncorrected_cum)
    corrected_daily_new = get_corrected_daily_new(uncorrected_daily_new)
    corrected_cumulative = corrected_daily_new.cumsum().dropna(how='any', axis='index')

    # sanity checks - cumulative totals should not have changed
    delta = 0.0001
    check = ( (uncorrected_cum.iloc[-1]-delta < corrected_cumulative.iloc[-1]) &
              (corrected_cumulative.iloc[-1] < uncorrected_cum.iloc[-1]+delta) )
    assert(check.all())
    check2 = ( ( (uncorrected_daily_new.sum() - delta) < corrected_daily_new.sum()) &
                 (corrected_daily_new.sum() < (uncorrected_daily_new.sum() + delta) ) )
    assert(check2.all())
    assert(len(uncorrected_cum) == len(corrected_cumulative))
    assert(len(uncorrected_daily_new) == len(corrected_daily_new))
  
    return (uncorrected_daily_new,
            corrected_daily_new,
            corrected_cumulative)


### --- Plotting
    
# matplotlib stuff for date formatting xticklabels
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

def start_point(country_name):
    if country_name == 'China':
        return pd.Timestamp('2019-12-30')
    if country_name in ['South Korea', 'Iran', 'France', 'Italy', 'Spain',
                        'San Marino', 'Australia',
                        
                        'ACT', 'NT', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']:
        return pd.Timestamp('2020-02-15')
    return pd.Timestamp('2020-03-01')


def _make_figure_adj(fig, kwargs):
    """ find keywords that relate to the figure,
        make the change, and return the kwargs
        dictionary, with these key words removed.
        Also return tight and savefig."""

    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig.set_size_inches(8, 4)
        
    if 'lfooter' in kwargs:
        fig.text(0.01, 0.01, kwargs['lfooter'],
             ha='left', va='bottom',
             fontsize=9, fontstyle='italic',
             color='#999999')
        del kwargs['lfooter']
        
    if 'rfooter' in kwargs:
        fig.text(0.99, 0.01, kwargs['rfooter'],
             ha='right', va='bottom',
             fontsize=9, fontstyle='italic',
             color='#999999')
        del kwargs['rfooter']
    
    tight = None
    if 'tight' in kwargs:
        tight = kwargs['tight']
        del kwargs['tight']
    else:
        tight = 1
        
    savefig = 'None'
    if 'savefig' in kwargs:
        savefig =kwargs['savefig']
        del kwargs['savefig']

    return kwargs, tight, savefig


def _annotate_bars_on_chart(series, ax):
    # annotate the plot
    span = series.max() - series.min()
    inside = series.min() + span / 2.0
    spacer = span / 150.0
    for y, x in enumerate(series):
        xpos = x if x < inside else 0
        color = 'black' if xpos > 0 else 'white'
        ax.annotate(f'{x}', xy=(xpos+spacer, y), 
                   va='center', color=color, size='small')


def plot_barh(series, **kwargs):
    """plot series as a horizontal bar chart"""
    
    fig, ax = plt.subplots()
    ax.barh(series.index, series, color='gray')
    _annotate_bars_on_chart(series, ax)

    kwargs, tight, savefig = _make_figure_adj(fig, kwargs)

    ax.margins(0.01)
    ax.set(**kwargs)
    fig.tight_layout(pad=tight)
    if savefig:
        fig.savefig(savefig, dpi=125)

    if SHOW: plt.show()
    plt.close()


# deal with large numbers
def label_maker(s: pd.Series, base_label: str)-> Tuple[pd.Series, str]:
    label = base_label
    if s.max() > 2_000_000:
        s /= 1_000_000 
        label += ' ($10^6$)'
    elif s.max() > 2_000:
        s /= 1_000
        label += " ($10^3$)"
    return s, label


def plot_orig_smooth(orig, n, mode, name, **kwargs):
    """plot series in original and smoothed forms,
       return the smoothed series"""
    
    orig, label = label_maker(orig, f'Daily new {mode}')
    if 'ylabel' in kwargs: kwargs['ylabel'] = label
    
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
    fig = ax.figure
    kwargs, tight, savefig = _make_figure_adj(fig, kwargs)
    ax.set(**kwargs)
    fig.tight_layout(pad=tight)
    
    # fudge for ax.margins(0.01) [which does not seem to work]
    xlim = ax.get_xlim()
    adj = (xlim[1] - xlim[0]) * 0.01
    ax.set_xlim(xlim[0]-adj, xlim[1]+adj)
    
    if savefig:
        fig.savefig(savefig, dpi=125)

    if SHOW: plt.show()
    plt.close()
    return hendo


def plot_growth_factor(new: pd.Series, **kwargs):
    """Week on week growth in new using a non-linear growth factor axis"""

    # calculate rolling average week-on-week growth factor    
    WEEK = 7 # days
    gf = new.rolling(WEEK).mean()
    gf_original = gf / gf.shift(WEEK)

    # adjusted scale
    gf = gf_original.where(gf_original<=1, other=2-(1/gf_original))

    # trims the start date
    start = start_point(new.name)
    gf = gf[gf.index >= start]

    # plot above and below 1 in different colours - resample to hourly to do this
    # this code is a bit of a hack
    gf1 = gf.resample('H').interpolate(method='linear', limit=23, limit_area="inside",
                                      limit_direction='forward')
    gf2 = gf.resample('H').interpolate(method='linear', limit=23, limit_area="inside",
                                      limit_direction='backward')
    gf = gf1.where(gf1.notna() & gf2.notna(), other=np.nan) # note data is now hourly
    below = gf.where(gf < 1, other=np.nan)                 # note data is now hourly
    
    # plot
    ax = gf.plot.line(lw=2, color='#B81D13', label='Growth (>1)')
    below.plot.line(lw=2, color='#008450', label='Decline (<1)', ax=ax)
    ax.axhline(1, lw=2, color='gray')
    ax.legend(loc='best', fontsize='small')
    ax.set_ylim(-0.05, 2.05)
    ax.set_yticks([0, 1/6, 1/3, 1/2, 3/4, 1, 2-3/4, 2-1/2, 2-1/3, 2-1/6, 2])
    ax.set_yticklabels(["0", "$\\frac{1}{6}$", "$\\frac{1}{3}$", 
                        "$\\frac{1}{2}$", "$\\frac{3}{4}$", "1", 
                        "$\\frac{4}{3}$", "2", "3", "6", "$\infty$"])
    if 'ylabel' in kwargs:
        kwargs['ylabel'] += ' (non-linear scale)'

    fig = ax.figure
    kwargs, tight, savefig = _make_figure_adj(fig, kwargs)
    ax.set(**kwargs)
    
    # fudge - because ax.margins(0.01) did not work
    xlim = ax.get_xlim()
    adj = (xlim[1] - xlim[0]) * 0.01
    ax.set_xlim(xlim[0]-adj, xlim[1]+adj)

    # put the latest growth factor on the plot
    right_annotation = ' ($GF_{end}=' f'{np.round(gf_original[-1], 2)}$)'
    fig.text(0.999, 0.15, right_annotation,
            rotation=90, ha='right', va='bottom', fontsize=9,
            color='#333333')

    # wrap up
    fig.tight_layout(pad=tight)
    if savefig:
        fig.savefig(savefig, dpi=125)

    if SHOW: plt.show()
    plt.close()


def plot_new_cum(new, cum, mode, name, **kwargs):
    """ derive new from cumulative series
    plot new as bars and cumulative s a line.
    Return the new series."""
    
    # adjust START
    start = start_point(name)
    new = new[new.index >= start]
    cum = cum[cum.index >= start]
    
    # adjust for large numbers
    cum, cum_label = label_maker(cum, f'Total {mode}')
    new, new_label = label_maker(new, f'New {mode}')
    
    # plot new
    MARGINS = 0.01
    fig, ax = plt.subplots()
    ax.xaxis_date()
    #ax.margins(MARGINS) # seems to work here

    ax.bar(new.index, new.values, 
               color='#dd0000', label=new_label)

    # plot cumulative
    axr = ax.twinx()
    axr.plot(cum.index, cum.values,
             lw=2.0, color='#0000dd', ls='--',
             label=cum_label)
    axr.set_ylabel(None)
    axr.grid(False)

    # put in a legend
    # let's assume that if len(new) < 100, we are doing recent
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    loc = 'lower left' if len(new) < 100 else 'upper left'
    axr.legend(h1+h2, l1+l2, loc=loc)

    # align the base of the left and right scales
    if (new >= 0).all():
        yliml = list(ax.get_ylim())
        yliml[0] = 0
        ax.set_ylim(yliml)
    if (cum >= 0).all():
        ylimr = list(axr.get_ylim())
        ylimr[0] = 0
        axr.set_ylim(ylimr)
    
    # Not sure why - but I need this
    if (new < 0).any():
        xlim = axr.get_xlim()
        ax.set_xlim(xlim)
    
    # This makes the dates for xticklabels nicer
    # let's assume that if len(new) < 100, we are doing recent
    locator = mdates.AutoDateLocator(minticks=4, maxticks=13)
    formatter1 = mdates.ConciseDateFormatter(locator)
    formatter2 = mdates.DateFormatter("%b-%d")
    formatter = formatter2 if len(new < 100) else formatter1
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # double check on those margins
    ax.margins(MARGINS)
    axr.margins(MARGINS)
    
    kwargs, tight, savefig = _make_figure_adj(fig, kwargs)
    ax.set(**kwargs)
    
    # y-axis labels - the hard way
    lHeight = 0.96
    lInstep = 0.02
    fig.text(1.0-lInstep, lHeight, cum_label,
            ha='right', va='top', fontsize=11,
            color='#333333')
    fig.text(lInstep, lHeight, new_label,
            ha='left', va='top', fontsize=11,
            color='#333333')

    fig.tight_layout(pad=tight)
    if savefig:
        fig.savefig(savefig, dpi=125)

    if SHOW: plt.show()
    plt.close()
    return None

#def _add_guides(ax, pos):
#    ylim = list(ax.get_ylim())
#    ylim[0] = min(0, ylim[0])
#    ylim[1] = max(50, ylim[1])
#    ax.set_ylim(ylim)
#    for i in [2, 3, 4, 7, 14]: # days
#        guide = ((2**(1/i)) - 1) * 100 # per cent
#        ax.axhline(guide, ls=':', color='black', lw=0.75)
#        ax.text(pos, guide, f'Doubles every {i} days',
#                ha='left', va='bottom', fontsize=9, 
#                fontstyle='italic', color='black')


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
            
    if 'savefig_prefix' in k_copy:
        prefix = k_copy['savefig_prefix']
        del k_copy['savefig_prefix']
    else:
        prefix = None
    
    generate_title = not 'title' in k_copy
    
    saved_k = copy.deepcopy(k_copy)
    for region, states in regions.items():
  
        # plot context in light grey
        ax = df.plot(c='#aaaaaa', lw=0.5)
        ax.get_legend().remove()
       
        if generate_title:
            k_copy['title'] = f'COVID-19 Daily New {mode.title()} - {region}'
        
        # plot this group in color
        subset = df[states]
        ax_new = ax.twinx()
        subset.plot(ax=ax_new, linewidth=3.5, legend=True)
        ax_new.legend(title=None, loc="upper left")
        ax_new.grid(False)
        ax_new.set_yticklabels([])
        ax_new.set_ylim(ax.get_ylim())
         
        # finalise
        fig = ax.figure
        k_copy, tight, savefig = _make_figure_adj(fig, k_copy)
        ax.set(**k_copy)
        fig.tight_layout(pad=tight)

        if prefix and generate_title:
            savefig = prefix + '!' + k_copy['title'] + '.png'
        if savefig:
            fig.savefig(savefig, dpi=125)

        if SHOW:
            plt.show()  
        plt.close()
        k_copy = copy.deepcopy(saved_k)
