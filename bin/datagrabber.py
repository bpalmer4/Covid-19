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
#
# Note: EU/ECDC switched to a weekly reporting schedule for the COVID-19 
# situation worldwide and in the EU/EEA and the UK on 17 December 2020

import numpy as np
import pandas as pd
import datetime
import urllib3
import re

_data_lake = {}
def get_OWID_data():
    URL = ('https://github.com/owid/covid-19-data/raw/master/'
           'public/data/owid-covid-data.csv')

    if URL in _data_lake:
        df =  _data_lake[URL].copy()
        print('Data retrieved from the lake')
    else:
        print(URL)
        df = pd.read_csv(URL, header=0)
        print('We have the data')
        _data_lake[URL] = df.copy()

    # get cases and deaths, remove null columns; make index a DatetimeIndex
    cases = df.pivot(columns='location', index='date', values='total_cases')
    cases = cases.dropna(axis='columns', how='all')
    source = f'OWID {cases.index[-1]}'
    cases.index = pd.DatetimeIndex(cases.index)
    deaths = df.pivot(columns='location', index='date', values='total_deaths')
    deaths = deaths.dropna(axis='columns', how='all')
    deaths.index = pd.DatetimeIndex(deaths.index)
    
    # population data for each country
    population = df.pivot(columns='location', index='date', values='population')
    populations = {}
    for p in population.columns:
        address = population[p].last_valid_index()
        if address is None:
            continue
        populations[p] = population[p].loc[address]
    
    # return the lot
    return cases, deaths, pd.Series(populations).astype(int), source

