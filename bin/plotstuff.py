# plot stuff - a series of functions to assist with (a) data ETL
#              and (b) plotting for COVID-19

# general python imports
from typing import List, Dict, Tuple, Set, Optional, Any, Iterable, Callable, Union
import datetime
import math
import copy
import sys
import csv
from re import match
from io import StringIO

# data science imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.ticker as ticker

import pandas as pd
import numpy as np

# local imports
from Henderson import Henderson


### --- data cleaning


def hollow_mean(series: pd.Series, middle: int = None) -> float:
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
    # assert(length >= 3 and length % 2)
    return (series.iloc[:middle].sum() + series[middle + 1 :].sum()) / (length - 1)


def rolling_mean_excluding_self(series: pd.Series, window: int = 15) -> pd.Series:
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

    positive = series.where(series >= 0, other=0)
    mean = positive.rolling(window, center=True).apply(hollow_mean)
    for n in range(int(window / 2) + 1, window):
        position = -(window - n)
        mean.iloc[position] = hollow_mean(positive[-window:], n)
    return mean


def replace_proportional(
    series: pd.Series, point: pd.Timestamp, replacement: float
) -> pd.Series:
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

    # try to limit any adjustments to the previous period of ...
    MAX_ADJUSTMENT_SPAN = 120  # days

    # sanity checks
    assert series.index.is_monotonic_increasing
    assert series.index.is_unique

    # replace original and remember the adjustment from the original
    adjustment = series[point] - replacement
    series.loc[point] = replacement

    # apply adjustment to this point and all preceeding data
    basement = point - pd.Timedelta(days=MAX_ADJUSTMENT_SPAN)
    base = series.index[0]
    if basement > base:
        available = series[(series.index >= basement) & (series.index < point)].sum()
        if adjustment > 0 or available > abs(adjustment):
            base = basement

    window = (series.index >= base) & (series.index <= point)
    assert (series[window] >= 0).all()  # a sanity check
    sum_for_window = series[window].sum()
    if sum_for_window > 0:
        factor = (sum_for_window + adjustment) / sum_for_window
        series[window] = series[window] * factor
    else:
        print(f"Warning: negative adjustment {series.name}")
        sys.exit(1)

    return series


def replace_nearby(
    series: pd.Series, point: pd.Timestamp, replacement: float = 0
) -> pd.Series:
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
    # assert(series.index.is_monotonic_increasing)
    # assert(series.index.is_unique)
    # assert(series[point] < 0)

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
    assert adjustment >= 0

    return series


def rolling_zero_count(series: pd.Series, window: int = 15) -> pd.Series:
    """ Count the number of zero or negative values nearby each 
        element in a series, where nerby is defined by the centered 
        window parameter."""

    return series.rolling(window, center=True).apply(lambda x: (x <= 0).sum()).ffill()


def negative_correct_daily(series: pd.Series, verbose) -> pd.Series:
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
            print(f"There are negatives in {series.name}")
            print(f"{series[negatives]}")
        replacers = rolling_mean_excluding_self(series).fillna(0)
        # make the adjustments
        for fix_me in negatives:
            if series[fix_me] >= MINOR_NEG:
                series = replace_nearby(series, fix_me)
            else:
                series = replace_proportional(
                    series, fix_me, replacement=replacers[fix_me]
                )
    return series


def positive_correct_daily(series: pd.Series, verbose) -> pd.Series:
    """Correct excess positive spikes in a series."""

    MIN_SPIKE = 30  # ignore smaller spikes
    SPIKE = 5  # defined as SPIKE times the local daily normal

    # consecutive data check - no point trying to deal with outliers in sparse data
    AT_LEAST_ONCE_CONSEC = 15  # n the series we have some consecutive data
    positive = (series >= 0.001).astype(int)
    id = positive.diff().ne(0).cumsum()
    max_consecutive = positive.groupby(id).cumsum().max()
    if max_consecutive < AT_LEAST_ONCE_CONSEC:
        if verbose:
            print(
                f"Data too sparse in {series.name} "
                f"(max_consecutive={max_consecutive})"
            )
        return series

    # identify adjustments
    # cumsum = series.cumsum()
    zeros = rolling_zero_count(series)
    rolling_mean = rolling_mean_excluding_self(series)
    spikes = (series > (SPIKE + (zeros * 2)) * rolling_mean) & (series > MIN_SPIKE)
    spikes = series[spikes].index

    # make changes
    original = series.copy()
    if len(spikes):
        spike_frame = pd.DataFrame(
            [series[spikes], rolling_mean[spikes], zeros[spikes]],
            index=["spike", "mean", "zeros"],
        )
        if verbose:
            print(f"Spikes in {series.name}\n{spike_frame}")

    for fix_me in spikes:
        series = replace_proportional(series, fix_me, replacement=rolling_mean[fix_me])

    # check nothing has gone wrong
    # final check - do no harm
    ACCEPTABLE_INCREASE = 1.075  # 7.5 per cent is the max acceptable increase
    if (
        (series.max() > (original.max() * ACCEPTABLE_INCREASE)) & (original >= 0)
    ).all():
        # we have not made things better
        if verbose:
            print(
                f"Spike not fixed for {series.name} "
                f"({series.max() / original.max()})"
            )
        series = original

    return series


def get_corrected_daily_new(input_frame: pd.DataFrame, verbose) -> pd.DataFrame:
    output_frame = pd.DataFrame(
        np.nan, columns=input_frame.columns, index=input_frame.index
    )
    for col in input_frame.columns:
        series = negative_correct_daily(input_frame[col], verbose)
        output_frame[col] = positive_correct_daily(series, verbose)
    return output_frame


def get_uncorrected_daily_new(cum_frame: pd.DataFrame) -> pd.DataFrame:

    # prepend a row of zeros - for the diff
    start = cum_frame.index.min()
    previous = start - pd.Timedelta(days=1)
    empty_row = pd.DataFrame(0, columns=cum_frame.columns, index=[previous])
    cum_frame = pd.concat([cum_frame, empty_row]).sort_index(ascending=True)

    # remove NAs, diff, drop the previously inserted row
    daily_frame = cum_frame.ffill().fillna(0).diff().drop(previous)
    return daily_frame


def dataframe_correction(
    uncorrected_cum: pd.DataFrame,
    corrections: Optional[pd.DataFrame] = None,
    verbose=True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Take an uncorrected dataframe of daily cumulative data,
        and return three dataframes:
        - uncorrected daily new data
        - corrected daily new data
        - corrected cumulative data"""

    nans = uncorrected_cum.isna()

    uncorrected_cum = uncorrected_cum.ffill().fillna(0)
    uncorrected_daily_new = get_uncorrected_daily_new(uncorrected_cum)
    corrected_daily_new = get_corrected_daily_new(uncorrected_daily_new, verbose)
    corrected_cumulative = corrected_daily_new.cumsum().dropna(how="all", axis="index")

    # sanity checks - cumulative totals should not have changed
    delta = 0.0001
    check = (uncorrected_cum.iloc[-1] - delta < corrected_cumulative.iloc[-1]) & (
        corrected_cumulative.iloc[-1] < uncorrected_cum.iloc[-1] + delta
    )
    if not check.all():
        print(
            f"WARNING: \n{uncorrected_cum.iloc[-1]}\n"
            f"{corrected_cumulative.iloc[-1]}\n"
            f"{uncorrected_daily_new.head(5)}\n"
            f"{corrected_daily_new.head(5)}"
        )
    check2 = ((uncorrected_daily_new.sum() - delta) < corrected_daily_new.sum()) & (
        corrected_daily_new.sum() < (uncorrected_daily_new.sum() + delta)
    )
    if not check2.all():
        print(
            "WARNING: \n{uncorrected_cum.iloc[-1]}\n" f"{corrected_cumulative.iloc[-1]}"
        )
    assert len(uncorrected_cum) == len(corrected_cumulative)
    assert len(uncorrected_daily_new) == len(corrected_daily_new)

    # restore nans
    for data in (uncorrected_daily_new, corrected_daily_new, corrected_cumulative):
        data.where(~nans, other=np.nan, inplace=True)

    return (uncorrected_daily_new, corrected_daily_new, corrected_cumulative)


### --- Plotting

# matplotlib stuff for date formatting xticklabels
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter
munits.registry[pd.Timestamp] = converter
munits.registry[pd.Period] = converter


# --- private

def get_selected_list(d:Dict[str, Any], selection:List[str], keep:bool=False
                ) -> Dict[str, Any]:
    """Return in a dictionary the items in the input dictionary d that
       are in the list selection. Do not change the original dictionary"""
    returnable = {key: d[key] for key in selection if key in d}
    for key in selection:
        if key in d and not keep:
            del d[key]
    return returnable


def get_selected_item(d:Dict[str, Any], key:str, default:Any=None, keep:bool=False) -> Any:
    """Return the item from the dictionary d with key, or return the default.
       Delete thie item from the dictionary if it was found there."""
    returnable = d[key] if key in d else default
    if key in d and not keep:
        del d[key]
    return returnable


def add_defaults(dictionary:Dict[str, Any], default_dict:Dict[str, Any]
                ) -> Dict[str, Any]:
    """Add default values to a dictionary, if they are not present in the 
       dictionary. Return the compilation as a new dictionary."""
    return {**default_dict, **dictionary}


def scatter(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]):
    """Add a scatter plot to an axes."""
    recent = get_selected_item(kwargs, key='recent', default=0)
    USE = ('c', 's', 'alpha', 'label')
    DEFAULTS = {'c': '#dd0000', 's': 10}
    params = add_defaults(get_selected_list(kwargs, USE), DEFAULTS)
    ax.scatter(series.index[-recent:], series.iloc[-recent:], **params)
    return None


def line(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]) -> None:
    """Add a line plot to a chart."""
    recent = get_selected_item(kwargs, key='recent', default=0)
    USE = ('color', 'c', 'alpha', 'label', 'linestyle', 'ls',
           'linewidth', 'lw', 'marker')
    param = get_selected_list(kwargs, USE)
    ax.plot(series.index[-recent:], series.iloc[-recent:], **param)
    return None


def bar(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]) -> None:
    """Add a bar plot to a chart."""
    recent = get_selected_item(kwargs, key='recent', default=0)
    USE = ('color', 'c', 'alpha', 'label', 'width')
    param = get_selected_list(kwargs, USE)
    ax.bar(series.index[-recent:], series.iloc[-recent:], **param)
    return None


def annotate_barh(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]) -> None:
    # annotate the plot
    round_ = get_selected_item(kwargs, key='round', default=None)
    span = series.max() - series.min()
    inside = series.min() + span / 2.0
    spacer = span / 150.0
    for y, x in enumerate(series):
        xpos = x if x < inside else 0
        color = "black" if xpos > 0 else "white"
        x = round(x, round_) if round_ else x
        ax.annotate(
            f"{x:,}", xy=(xpos + spacer, y), 
            va="center", color=color, size="small"
        )
    return None


def barh(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]) -> None:
    """Add a horizontal bar chart."""
    USE = ('color', 'c', 'alpha', 'label', 'width')
    param = get_selected_list(kwargs, USE)
    ax.barh(series.index.values, series.values, **param)
    annotate_barh(ax, series, kwargs)
    return None


def multi_ma(ax:plt.axes, series:pd.Series, kwargs:Dict[str, Any]) -> None:
    """Add one or more moving averages to a chart."""
    
    periods = get_selected_item(kwargs, key='periods', default=[7, 14])
    colors = get_selected_item(kwargs, key='colors', default=['dodgerblue', 'navy'])
    linestyles = get_selected_item(kwargs, key='linestyles', default=['-', '-'])
    linewidths = get_selected_item(kwargs, key='linewidths', default=[1.5, 2.5])
    
    for i, period in enumerate(periods):
        ma = series.rolling(period, center=True).mean()
        arg_dict = kwargs.copy()
        arg_dict['c'] = colors[i%len(colors)]
        arg_dict['ls'] = linestyles[i%len(linestyles)]
        arg_dict['lw'] = linewidths[i%len(linewidths)]
        arg_dict['label'] = f'{period}-day centred moving average'
        line(ax, ma, {**kwargs, **arg_dict})
    return None


SCALES = {
    0: '',
    3: 'thousand', 6: 'million', 9: 'billion',
    12: 'trillion', 15: 'quadrillion', 18: 'quintillion',
}

def scale_series(series:pd.Series, kwargs:Dict[str, Any]) -> Tuple[pd.Series, int, str]:
    """Scale a series, if 'scale_y'=True in kwargs. Also adjust ylabel text
       if that is present in kwargs. Factor is as in 10 ** factor. 
       Returns a tuple: adjusted series, factor, factor-text."""
    
    # do we need to act
    scale_y = get_selected_item(kwargs, 'scale_y', default=False)
    scale_x = get_selected_item(kwargs, 'scale_x', default=False)
    if not scale_x and not scale_y:
        return series, 0, ''
    
    label = 'xlabel' if scale_x else 'ylabel'
    max = series.abs().max()
    factor = 0 if max < 1000 else np.floor(np.log10(max) / 3.0) * 3
    if factor > 0:
        if label not in kwargs:
            kwargs[label] = f'{SCALES[factor].title()}'
        else:
            kwargs[label] = f'{kwargs[label]} ({SCALES[factor]})'
    return (series / (10 ** factor), factor, SCALES[factor])


def _get_day(series):
    # find the last day of a pandas Series or DataFrame
    last_day = series.dropna().index[-1]  # last day with data
    return {
        0: "W-MON",
        1: "W-TUE",
        2: "W-WED",
        3: "W-THU",
        4: "W-FRI",
        5: "W-SAT",
        6: "W-SUN",
    }[last_day.dayofweek]


def adjust_ylim(ax, series) -> None:
    # Note: assume a zero minimum.
    MARGIN = 0.02
    maxi = series.max()
    ax.set_ylim([0, maxi*(1+MARGIN)])
    return None

    
def adjust_xlim(ax, axr, cum):
    MARGIN = 0.02
    mini = cum.index.min()
    maxi = cum.index.max()
    span = (maxi - mini) / pd.Timedelta(days=1)
    adjustment = pd.Timedelta(days=(span / 2) * MARGIN)
    ax.set_xlim([mini-adjustment, maxi+adjustment])
    return None


def get_comma_sep_pairs(s:str, type_convert_numbers=True) -> Dict[str,Union[str,int,float]]:
    """Split string into comma separated key=value pairs, return as a dictionary.
       Type convert numbers to int or float if type_convert_numbers is set to True."""
    
    pairs = csv.reader(StringIO(s))
    results={}
    for items in pairs:
        for p in items:
            k, v = p.split('=', maxsplit=1)
            if type_convert_numbers:
                if match(r'^-?\d+$', v):
                    v = int(v)
                elif match(r'^-?\d+(?:\.\d+)$', v):
                    v = float(v)
            results[k] = v
    return results


# --- public 

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
            x
            , yscale, margin
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
    DEFAULT_SET_SIZE_INCHES = (9, 4.5)
    DEFAULT_SAVE_TYPE = "png"
    DEFAULT_TIGHT_PADDING = 1.2
    DEFAULT_SAVE_TAG = ""
    AXES_SETABLE = (
        "title",
        "xlabel",
        "ylabel",
        "xticks",
        "yticks",
        "xticklabels",
        "yticklabels",
        "xlim",
        "ylim",
        "xscale",
        "yscale",
    )
    OTHER_SETABLE = (
        "lfooter",
        "rfooter",
        "tight_layout_pad",
        "set_size_inches",
        "save_as",
        "chart_directory",
        "save_type",
        "save_tag",
        "show",
        "display",
        "dont_close",
        "margins",
        "no_locator",
        "axhline",
    )
    IGNORE = {
        'recent',
    }

    # utility
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    # precautionary
    if "title" not in kwargs:
        eprint(
            "Warning: the plot title has not been set\n"
            "\tin the call to finalise_plot()."
        )
    for arg in kwargs:
        if arg not in AXES_SETABLE and arg not in OTHER_SETABLE and arg not in IGNORE:
            eprint(
                f'Warning: the argument "{arg}" in the call\n'
                "\tto finalise_plot() is not recognised."
            )

    # usual axes settings
    axes_settings = {}
    for arg in kwargs:
        if arg not in AXES_SETABLE:
            continue
        axes_settings[arg] = kwargs[arg]
    if len(axes_settings):
        ax.set(**axes_settings)

    # margins
    if "margins" in kwargs and kwargs["margins"] is not None:
        ax.margins(kwargs["margins"])
        
    # reference line
    if "axhline" in kwargs:
        keywords = get_comma_sep_pairs(kwargs['axhline'])
        ax.axhline(**keywords) # Note: need to specify y=...

    # increase y-axis locator
    if ax.get_yaxis().get_scale() == "linear" and "no_locator" not in kwargs:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(11))

    # right footnote
    fig = ax.figure
    if "rfooter" in kwargs and kwargs["rfooter"] is not None:
        fig.text(
            0.99,
            0.005,
            kwargs["rfooter"],
            ha="right",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            color="#999999",
        )

    # left footnote
    if "lfooter" in kwargs and kwargs["lfooter"] is not None:
        fig.text(
            0.01,
            0.005,
            kwargs["lfooter"],
            ha="left",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            color="#999999",
        )

    # figure size
    if "set_size_inches" in kwargs:
        size = kwargs["set_size_inches"]
    else:
        size = DEFAULT_SET_SIZE_INCHES
    fig.set_size_inches(*size)

    # tight layout
    if "tight_layout_pad" in kwargs:
        pad = kwargs["tight_layout_pad"]
    else:
        pad = DEFAULT_TIGHT_PADDING
    fig.tight_layout(pad=pad)

    # save the plot to file
    # - save using the specified file name
    save_as = None
    if "save_as" in kwargs:
        save_as = kwargs["save_as"]

    # - save using a file name built from diretory-title-tag-type
    elif "chart_directory" in kwargs:
        save_type = DEFAULT_SAVE_TYPE
        if "save_type" in kwargs:
            save_type = kwargs["save_type"]
        save_tag = DEFAULT_SAVE_TAG
        if "save_tag" in kwargs:
            save_tag = kwargs["save_tag"]
        # file-system safe
        if "title" in kwargs:
            title = kwargs["title"].replace("[:/]", "-")
        else:
            title = ""
        save_as = f'{kwargs["chart_directory"]}{title}' f"{save_tag}.{save_type}"

    # - warn if there is no saving arrangement
    else:
        eprint(
            "Warning: in the call to finalise_plot()\n"
            "\tyou need to specify either save_as or\n"
            "\tchart_directory to save a plot to file."
        )

    if save_as:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        fig.savefig(save_as, dpi=300)

    # show the plot
    if "show" in kwargs and kwargs["show"]:
        plt.show()

    # display the plot (from Jupyter Notebook)
    if "display" in kwargs and kwargs["display"]:
        display(fig)

    # close the plot
    if "dont_close" not in kwargs or not kwargs["dont_close"]:
        plt.close()

    return None


def plot_series_with_ma(series, **kwargs) -> None:
    """Plot recent daily incidence data onto a scatter plot or bar chart 
       with moving averages."""

    # Some standard arguemnts
    recent = get_selected_item(d=kwargs, key='recent', default=0, keep=True)
    trunc_series = series[-recent:]
    if trunc_series.isna().all() or trunc_series.sum() == 0:
        return None
    lfooter = f'Total in period: {int(trunc_series.sum()):,}'
    
    # data scaling ...
    series, factor, f_text = scale_series(series, kwargs)
    
    # individual plot
    plot_type = get_selected_item(kwargs, key='plot_type', default='scatter')
    fig, ax = plt.subplots()
    kwargs_copy = kwargs.copy()
    if plot_type == 'bar':
        bar(ax, series, kwargs_copy)
    else:
        scatter(ax, series, kwargs_copy)
    
    # plot moving averages
    multi_ma(ax, series, kwargs)

    # finalise
    ax.legend(loc='best')
    finalise_plot(ax, **kwargs)
    return None


def plot_weekly(series:pd.Series, **kwargs):
    
    recent = get_selected_item(d=kwargs, key='recent', default=0)
    daily = series[-recent:].dropna()
    
    rule = _get_day(series)
    weekly = daily.resample(rule=rule, closed="right").sum()
    max_weekly = weekly.max()
    last_week = weekly.iloc[-1]
    weekly.index = weekly.index - pd.Timedelta(3.5, unit="d")
    cum = daily.cumsum()
    max_cum = cum.iloc[-1]
    
    if max_cum <= 0:
        print('Nothing to see here')
        return None
    
    kwargs['title'] = '' if 'title' not in kwargs else f'Weekly New {kwargs["title"]}'
    wkwargs = kwargs.copy()
    wkwargs['width'] = 5
    wkwargs['color'] = '#cc0000'
    wkwargs['ylabel'] = '' if 'ylabel' not in wkwargs else f'Weekly New {wkwargs["ylabel"]}'
    weekly, wfactor, wf_text = scale_series(weekly, wkwargs)
    wkwargs['label'] = wkwargs['ylabel']
    
    ckwargs = kwargs.copy()
    ckwargs['color'] = 'navy'
    ckwargs['lw'] = '2.5'
    ckwargs['ls'] = '-.'
    ckwargs['ylabel'] = '' if 'ylabel' not in ckwargs else f'Cum. {ckwargs["ylabel"]}'
    cum, cfactor, cf_text = scale_series(cum, ckwargs)
    right_ylabel = ckwargs['ylabel']
    ckwargs['label'] = ckwargs['ylabel']

    # plot
    fig, ax = plt.subplots()
    bar(ax, weekly, wkwargs)
    axr = ax.twinx()
    line(axr, cum, ckwargs)
    
    # adjustments because I am twinning the x axis
    axr.set_ylabel(right_ylabel)
    axr.grid(False)
    adjust_ylim(ax, weekly)
    adjust_ylim(axr, cum)
    adjust_xlim(ax, axr, cum)
    
    # legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")
    
    wkwargs['lfooter'] = (
        f'WE={rule[-3:].title()}; '
        f'Max weekly={int(max_weekly):,}; '
        f'Over the past week={int(last_week):,}; '
        f'Cum. total={int(max_cum):,}'
    )
    
    finalise_plot(ax, **wkwargs)
    return None


def loop_over_frame(df:pd.DataFrame, desc:str, func:Callable, **kwargs):
    
    lfooter = False
    if 'lfoot_series' in kwargs:
        lfoot_series = get_selected_item(kwargs, key='lfoot_series', default=None)
        lfoot_text = get_selected_item(kwargs, key='lfoot_text', default='{}')
        lfooter = True
        
    for name in df.columns:
        #print(name)
        kwargs_copy = kwargs.copy()
        series = df[name]
        kwargs_copy['title'] = f'{desc} - {name}' if 'title' not in kwargs_copy else kwargs_copy['title']
        kwargs_copy['ylabel'] = desc if 'ylabel' not in kwargs_copy else kwargs_copy['ylabel']
        if lfooter:
            kwargs_copy['lfooter'] = lfoot_text.format(lfoot_series[name])
        
        func(
            series,
            **kwargs_copy,
        )
        
    return None


def plot_final_barh(df, **kwargs):
    
    MAXIMUM = 40
    
    # get latest values
    series = df.ffill().iloc[-1]
    last_valid = df.apply(pd.Series.last_valid_index)
    labels = (
        # Hint: if this is printing rge day as a float, 
        # then you have NaT in yout index. 
        last_valid.dt.day.astype(str) 
        + "-" 
        + last_valid.dt.month_name().astype(str).str[:3]
    )
    series.index = series.index.astype(str) + ' ' + labels
    
    if len(series) > MAXIMUM:
        series = series.sort_values(ascending=False).iloc[:MAXIMUM]
    
    # scale and sort the data series
    series, factor, f_text = scale_series(series, kwargs)
    force_int = get_selected_item(kwargs, key='force_int', default=False)
    if force_int and not factor: 
        series = series.astype(int)
    series = series.sort_values().copy()
    
    # plot
    fig, ax = plt.subplots()
    barh(ax, series, kwargs)
    finalise_plot(ax, **kwargs)
    return None


def plot_multiline(df, **kwargs):
    color_dict = get_selected_item(kwargs, key='color_dict', default=None)
    USE = ('alpha', 'linestyle', 'ls',
           'linewidth', 'lw', 'marker',)
    param = get_selected_list(kwargs, USE)
    param['recent'] = get_selected_item(kwargs, key='recent', default=0)
    
    fig, ax = plt.subplots()
    for col in df.columns:
        p = param.copy()
        if color_dict:
            p['c'] = color_dict[col]
        p['label'] = col
        line(ax, df[col], p)
    legend = {'loc': 'best', }
    if 'legend_ncol' in kwargs:
        legend['ncol'] = get_selected_item(kwargs, key='legend_ncol', default=1)
    ax.legend(**legend)
    finalise_plot(ax, **kwargs)    
    return None


def daily_growth_rate(series:pd.Series, **kwargs):
    PERIOD = 7
    THRESHOLD = 10 # minimum cases per day on average
    
    # growth rate
    series = series.rolling(PERIOD).mean()
    series = series.where(series >= THRESHOLD, other=np.nan) # ignore small data
    k = np.log(series / series.shift(PERIOD)) / PERIOD  * 100 # daily growth rate %
    if k.isna().all():
        return None
    fig, ax = plt.subplots()
    line(ax, k, kwargs)
    ax.axhline(0, color='#999999', lw=0.5)
    previous_lfooter = kwargs['lfooter'] if 'lfooter' in kwargs else ''
    kwargs['lfooter'] = f'When daily new cases >= {THRESHOLD}; ' + previous_lfooter
    finalise_plot(ax, **kwargs)
    return None


def five_day_on_five_day(series:pd.Series, **kwargs):
    PERIOD = 5
    THRESHOLD = 20 # minimum cases per day on average
    
    # growth rate
    series = series.rolling(PERIOD).mean()
    series = series.where(series >= THRESHOLD, other=np.nan) # ignore small data
    growth = series / series.shift(PERIOD)

    # plot
    fig, ax = plt.subplots()
    line(ax, growth, kwargs)
    ax.axhline(1, color='#999999', lw=0.5)
    previous_lfooter = kwargs['lfooter'] if 'lfooter' in kwargs else ''
    kwargs['lfooter'] = f'When daily new cases >= {THRESHOLD}; ' + previous_lfooter
    finalise_plot(ax, **kwargs)
    return None


def short_run_projection(series, **kwargs):
    PERIODS = [3, 7] # days 
    COLORS = ['#444444', 'darkred']
    SMOOTH_TERM = 15 # days
    smooth = Henderson(series.dropna(), SMOOTH_TERM)
    PROJECT = 7
    
    fig, ax = plt.subplots()
    kw = kwargs.copy()
    kw['label'] = 'Daily new cases'
    kw['c'] = 'royalblue'
    kw['lw'] = 1
    line(ax, series, kw)
    
    kw = kwargs.copy()
    kw['label'] = f'Smoothed {SMOOTH_TERM}-term HMA'
    kw['c'] = 'darkorange'
    kw['lw'] = 2
    line(ax, smooth, kw)
    
    for i, p in enumerate(PERIODS):
        x0 = smooth[-p-1]
        xt = smooth[-1]
        if x0 <= 0 or xt <= 0:
            # avoid the problematic
            continue
        k = np.log(xt / x0) / p
        a = smooth[-1] 
        t = pd.Series(range(0, PROJECT+1), 
                      index=pd.date_range(start=series.index[-1], 
                                          periods=PROJECT+1))
        projection = a * np.exp(k * t)
        
        kw = kwargs.copy()
        kw['label'] = f'Projection based on recent {p}-day growth'
        kw['c'] = COLORS[i]
        kw['lw'] = 2.5
        kw['ls'] = ':'
        line(ax, projection[1:], kw)
        
    ax.legend(loc='best')
    finalise_plot(ax, **kwargs)
