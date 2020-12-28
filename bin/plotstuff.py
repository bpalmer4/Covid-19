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

### --- data cleaning

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
    return series.rolling(WINDOW, center=True).apply(
        lambda x: (x <= 0).sum()).ffill()


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
            series = replace_proportional(series, fix_me, 
                                          replacement=rolling_mean[fix_me])
    return series


def positive_correct_daily(series: pd.Series)-> pd.Series:
    """Correct excess positive spikes in a series."""

    MIN_SPIKE = 30 # ignore smaller spikes
    SPIKE = 5 # defined as SPIKE times the local daily normal

    # consecutive data check - no point trying to deal with outliers in sparse data
    AT_LEAST_ONCE_CONSEC = 15 # n the series we have some consecutive data
    positive = (series >= 0.001).astype(int)
    id = positive.diff().ne(0).cumsum()
    max_consecutive = positive.groupby(id).cumsum().max()
    if max_consecutive < AT_LEAST_ONCE_CONSEC:
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
        print(f'Spikes in {series.name}\n{spike_frame}')
        
    for fix_me in spikes:
        series = replace_proportional(series, fix_me, 
                                      replacement=rolling_mean[fix_me])

    # check nothing has gone wrong
    # final check - do no harm
    ACCEPTABLE_INCREASE = 1.075 # 7.5 per cent is the Max acceptable increase
    if ((series.max() > (original.max() * ACCEPTABLE_INCREASE)) 
        & (original >= 0)).all():
        # we have not made things better
        print(f'Spike not fixed for {series.name} '
              f'({series.max() / original.max()})')
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
    corrected_cumulative = corrected_daily_new.cumsum().dropna(
        how='all', axis='index')

    # sanity checks - cumulative totals should not have changed
    delta = 0.0001
    check = ( (uncorrected_cum.iloc[-1]-delta < corrected_cumulative.iloc[-1]) &
              (corrected_cumulative.iloc[-1] < uncorrected_cum.iloc[-1]+delta) )
    assert(check.all())
    check2 = (((uncorrected_daily_new.sum() - delta) < corrected_daily_new.sum())
              & (corrected_daily_new.sum() < (uncorrected_daily_new.sum() + delta)))
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


def finalise_plot(ax, **kwargs):
    """A function to automate the completion of a plot, 
       including saving it to file and closing the plot when done.
       
       Arguments:
       - ax - required - a matplotlib axes object
       - title - required - string - the title to appear on the plot
       - matplotlib axes settings - optional - any/all of the 
         following : 
            title, xlabel, ylabel, xticks, yticks, 
            xticklabels, yticklabels, xlim, ylim, 
            xscale, yscale, 
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
                    'xscale', 'yscale',)
    OTHER_SETABLE = ('lfooter', 'rfooter', 'tight_layout_pad', 
                     'set_size_inches', 'save_as', 'chart_directory',
                     'save_type', 'save_tag', 'show', 'dont_close')
    
    # precautionary
    if 'title' not in kwargs:
        kwargs['title'] = 'Unknown title'
    for arg in kwargs:
        if arg not in AXES_SETABLE and arg not in OTHER_SETABLE:
            print(f'Warning: argument {arg} in call to '
                  'finalise_plot not recognised')
    
    # usual settings
    settings = {}
    for arg in kwargs:
        if arg not in AXES_SETABLE:
            continue
        settings[arg] = kwargs[arg]
    if len(settings):
        ax.set(**settings)
    
    fig = ax.figure
    
    # left footnote
    if 'rfooter' in kwargs and kwargs['rfooter'] is not None:
        fig.text(0.99, 0.005, kwargs['rfooter'],
            ha='right', va='bottom',
            fontsize=9, fontstyle='italic',
            color='#999999')
    
    # right footnote
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
    #print(size)
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
        title = kwargs['title'].replace('[:/]', '-')
        save_as = (f'{kwargs["chart_directory"]}{title}'
                   f'{save_tag}.{save_type}')

    # - warn if there is no saving arrangement
    else:
        print('Warning: You need to sepcify either save_as '
              'or chart_directory to save a plot to file.\n')

    if save_as:
        fig.savefig(save_as, dpi=125)
    
    # show the plot
    if 'show' in kwargs and kwargs['show']:
        plt.show()

    # close the plot
    if 'dont_close' not in kwargs or not kwargs['dont_close']:
        plt.close('all')
    
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
    numerical = ('', '$10^3$', '$10^6$', '$10^9$', '$10^12$', '$10^16$')
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


def plot_growth_factor(new_: pd.Series, **kwargs):
    """Week on week growth in new_ using a non-linear growth factor axis
       Uses the same **kwargs as finalise_plot()
       Note: if rfooter in **kwargs, it will be over-written"""

    # calculate rolling average week-on-week growth factor    
    WEEK = 7 # days
    gf = new_.rolling(WEEK).mean()
    gf_original = gf / gf.shift(WEEK)
    
    # adjusted scale
    gf = gf_original.where(gf_original<=1, other=2-(1/gf_original))

    # trims the start date
    start = start_point(new_.name)
    gf = gf[gf.index >= start]

    # plot above and below 1 in different colours 
    # - resample to hourly to do this
    # this code is a bit of a hack
    gf1 = gf.resample('H').interpolate(method='linear', limit=23, 
                                       limit_area="inside",
                                      limit_direction='forward')
    gf2 = gf.resample('H').interpolate(method='linear', limit=23, 
                                       limit_area="inside",
                                      limit_direction='backward')
    gf = gf1.where(gf1.notna() & gf2.notna(), other=np.nan) 
    below = gf.where(gf < 1, other=np.nan)                 
    
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
        kwargs['ylabel'] += '\n(non-linear scale)'

    xlim = ax.get_xlim()
    adj = (xlim[1] - xlim[0]) * 0.01
    ax.set_xlim(xlim[0]-adj, xlim[1]+adj)
    kwargs['rfooter'] = ' ($GF_{end}=' f'{np.round(gf_original[-1], 2)}$)'
    finalise_plot(ax, **kwargs)

    
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
    ax.margins(0.01)
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
    ax.legend(h1+h2, l1+l2, loc='upper left', fontsize='small')

    # adjust y-limits to be prettier, 
    # assume ylim[0] is zero, but also check
    # this adjustment should not be needed, but it is
    ylim = ax.get_ylim()
    ylim = ylim[0], ylim[1] * 1.025
    if ylim[0] != 0:
        # this should not happen - ever.
        print(f'Warning: ylim[0] is {ylim[0]} for {name}')
    ylimr = axr.get_ylim()
    axr.set_ylim((0, ylimr[1]))
    
    # Not sure why - but I need this
    #if (new < 0).any():
    #    xlim = axr.get_xlim()
    #    ax.set_xlim(xlim)

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


def plot_weekly(daily, mode, data_quality, dfrom="2020-01-21", **kwargs):
    """Plot weekly bar charts for daily new cases and deaths
        Function paramters:
        - daily is a DataFrame of daily timeseries data
        - mode is one of 'cases' or 'deaths' 
        - data_quality is a Series of strings,
            used for the left footnote on charts
        - dfrom is a date string to display from
        Returns: weekly data in a DataFrame """
    
    DISPLAY_FROM = pd.Timestamp(dfrom)
    
    # find the day that the week ends - last day of dataframe
    LAST_DAY = daily.index[-1]
    RULE = {
        0: 'W-MON',
        1: 'W-TUE',
        2: 'W-WED',
        3: 'W-THU',
        4: 'W-FRI',
        5: 'W-SAT',
        6: 'W-SUN',
    }[LAST_DAY.dayofweek]

    # convert the data to weekly
    returnable = weekly = daily.resample(rule=RULE, 
                                         closed='right').sum()
    total = weekly.sum()
    cum_weekly = weekly.cumsum()
    
    # we move the data by half a week becuase 
    # we want the bars to be centred
    weekly.index = weekly.index - pd.Timedelta(3.5, unit='d')
    cum_weekly.index = cum_weekly.index - pd.Timedelta(3.5, unit='d')
    
    for name in weekly.columns:
    
        # avoid plotting an empty plot
        if total[name] == 0: continue

        plot_new_cum(
            weekly[name].copy(), 
            cum_weekly[name].copy(), 
            mode, 
            name,
            'week',
            dfrom, 
            title=f'{name}: COVID-19 {mode.title()}',
            rfooter=data_quality[name],
            lfooter=f'Total {mode.lower()}: '
                    f'{int(total[name]):,}; '
                    f'(WE={RULE[-3:].title()})',
            **kwargs,
        )
        
    return returnable

    
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
