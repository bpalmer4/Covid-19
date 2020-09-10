# datagrabber.py
#
# Functions to get the data from one of two sources:
#
# * Our World in Data (OWID). For more information: 
#   https://ourworldindata.org/covid-sources-comparison
#
# * European Centre for Disease Prevention and Control (EU): 
#   https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
#
# Note: From 18 March OWID aligned its data with the EU
# And the two sources are much closer to each other than previously

import numpy as np
import pandas as pd
import datetime
import re

_data_lake = {}
def _get_data(url):
    if url in _data_lake:
        df =  _data_lake[url].copy()
        print('Data retrieved from the lake')
    else:
        print(f'Retrieving data from: {url}')
        csv = re.compile('\.[cC][sS][vV]$')
        if re.search(csv, url):
            df = pd.read_csv(url, header=0, 
                             index_col=0)
        else:
            print(url)
            df = pd.read_excel(url, header=0, 
                               index_col=0)
        _data_lake[url] = df.copy()
    return df

def get_data_from_owid(data_type):
    """get data from Our World In Data (OWID)
    https://ourworldindata.org/coronavirus-source-data"""
    
    # get the data
    url_map = {
        'cases': 'total_cases.csv',
        'deaths': 'total_deaths.csv'
    }
    url = ('https://covid.ourworldindata.org/data/ecdc/' +
            url_map[data_type])
    owid_total_cases = _get_data(url)
    owid_total_cases.index = (pd.Series(owid_total_cases.index)
        .astype('datetime64[ns]').dt.date)

                   
    # clean the data
    for check in ['World', 'International']:
        if check in owid_total_cases.columns:
            del owid_total_cases[check]
    owid_total_cases.columns = (
        owid_total_cases.columns.str.title().str.strip()
    )
    owid_total_cases = owid_total_cases.sort_index()
        
    # and use this one
    date = owid_total_cases.index[-1]
    source = f'OWID {date}'
    
    # index as Timestamps
    owid_total_cases.index = pd.DatetimeIndex(owid_total_cases.index)
    
    return owid_total_cases, source 

def _get_eu_http_addr():
    today = datetime.date.today()
    yesterday = (today - 
                 datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    url = (
        'https://www.ecdc.europa.eu/sites/default/files/documents/' +
        f'COVID-19-geographic-disbtribution-worldwide-{yesterday}.xlsx'    
    )
    return(url)

def _eu_column_cleaning(df):
    # some cleaning before the pivot
    #print('EU columns: ', df.columns.to_list())
    #country_col = 'Countries and territories'
    country_col = 'countriesAndTerritories'
    df[country_col] = (df[country_col]
                        .str.replace('_', ' ')
                        .str.title()
                        .str.strip())
    return(df, country_col)

_eu_new_names = {
        # primarily to match-up with OWID names
        'Brunei Darussalam': 'Brunei',
        'Democratic Republic Of The Congo': 
                'Democratic Republic Of Congo',
        'Holy See': 'Vatican', 
        'United States Of America': 'United States',
        'Cote Divoire': "Cote D'Ivoire",
}

def get_population_from_eu():
    url = _get_eu_http_addr()
    df = _get_data(url)
    df, country_col = _eu_column_cleaning(df)
    pivot = df.pivot_table(values='popData2019', 
                           index=country_col, 
                           aggfunc='first')
    # Note: pivot is a single column dataframe 
    #pivot = pivot[del eu_total_cases['Cases On An International Conveyance Japan']del eu_total_cases['Cases On An International Conveyance Japan']pivot.index != 'Cases On An International Conveyance Japan']
    pivot = pivot.rename(_eu_new_names)
    return pivot

def get_data_from_eu(data_type):
    """ Get data from the EU - https://www.ecdc.europa.eu/en
    Note: this data is in long form, comprising new cases
    and new deaths by date and country. We will transform
    to a wide form of cumulative cases similar to OWID data.
    """

    url = _get_eu_http_addr()
    df = _get_data(url)
    df, country_col = _eu_column_cleaning(df)

    # pivot into the form we want and cumsum
    data_type_map = {
        'cases': 'cases', #'Cases',
        'deaths': 'deaths' #'Deaths',
    }
    df['DateRep'] = pd.to_datetime(df.year*10000 + 
        df.month*100 + df.day, format='%Y%m%d')
    pivot = pd.pivot_table(df, 
                           values=data_type_map[data_type], 
                           index='DateRep', 
                           columns=country_col, 
                           aggfunc=np.sum)
    pivot = pivot.fillna(0.0)
    eu_total_cases = pivot.cumsum().sort_index().replace(0.0, np.nan)
    #print('EU countries: ', eu_total_cases.columns.to_list())
    
    #clean the data further
    #del eu_total_cases['Cases On An International Conveyance Japan']
    eu_total_cases = eu_total_cases.rename(columns=_eu_new_names)
    eu_total_cases = eu_total_cases.sort_index()
    
    # and use this one.
    date = eu_total_cases.index[-1].date()
    source = f'EU {date}'
    return(eu_total_cases, source)

def get_data(data_type, from_where):
    data = None
    source = 'Unsourced'

    if 'OWID' == from_where:
        owid, source = get_data_from_owid(data_type)
        data = owid

    if 'EU' == from_where:
        eu, source = get_data_from_eu(data_type)
        data = eu
    
    #print('============')
    #print(data.index[0])
    #print(data.index.dtype)
    #print(type(data.index[0]))
    #print('============')

    return data, source