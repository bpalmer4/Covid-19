## Henderson.py
## calculate a Henderson moving average

import pandas as pd
import numpy as np

def hmaSymmetricWeights(n):
    """ derive an n-term array of symmetric 'Henderson Moving Average' weights
        formula from ABS (2003), 'A Guide to Interpreting Time Series', page 41.
        returns a numpy array of symmetric Henderson weights indexed from 0 to n-1"""

    # calculate the constant denominator and terms
    m = int((n-1)//2) # the mid point - n must be odd
    m1 = (m+1)*(m+1)
    m2 = (m+2)*(m+2)
    d = float(8*(m+2)*(m2-1)*(4*m2-1)*(4*m2-9)*(4*m2-25))
    m3 = (m+3)*(m+3)

    # calculate the weights
    w = np.repeat(np.nan, n) # Actually indexed from 0 to n-1
    for j in range(m+1):
        j2 = j*j
        v = (315*(m1-j2)*(m2-j2)*(m3-j2)*(3*m2-11*j2-16))/d
        w[(m+j)] = v
        if j > 0:
            w[(m-j)] = v

    w.flags.writeable = False # let's make it quasi-immutable
    return (w)


def hmaAsymmetricWeights(m, w):
    """calculate the asymmetric end-weights

        w --> an array of symmetrical henderson weights (from above function)
        m --> the number of asymmetric weights sought; where m < len(w);

        returns a numpy array of asymmetrical weights, indexed from 0 to m-1;

        formula from Mike Doherty (2001), 'The Surrogate Henderson Filters in X-11',
        Aust, NZ J of Stat. 43(4), 2001, pp901-999; see formula (1) on page 903"""


    n = len(w) # the number of weights

    # - some quick sanity checks
    if m >= n:
        raise ValueError('The m argument must be less than n')
    if m <= int((n-1)//2):
        raise ValueError('The m argument must be greater than (n-1)/2')

    # --- let's build up Doherty's formula (1) from the top of page 903

    # - the second chunk of the formula
    sumResidual = w[range(m, n)].sum() / float(m)

    # - the last chunk of the formula
    sumEnd = 0.0
    for i in range(m+1, n+1):
        sumEnd += (float(i)-((m+1.0)/2.0)) * w[i-1] # w indexed from 0 to n-1

    # - the beta squared / sigma squared - formula at the bottom of page 904
    ic = 1.0
    if n >= 13 and n < 15:
        ic = 3.5
    elif n >= 15:
	    ic = 4.5
    b2s2 = (4.0/np.pi)/(ic*ic)

    # - the gnarly bit in the middle of the formula
    denominator = 1.0 + ((m*(m-1.0)*(m+1.0) / 12.0 ) * b2s2)
    u = np.repeat(np.nan, m) # return series - created empty
    for r in range(m): # r ranges 0 to m-1; but the formulae assumes 1 to m
        numerator = ((r+1.0) - (m+1.0)/2.0) * b2s2
        # - finally putting it all together
        u[r] = w[r] + sumResidual + ( numerator / denominator ) * sumEnd

    u.flags.writeable = False # let's make it quasi-immutable
    return (u)

def Henderson(s, n):
    """ Calculate an n-term Henderson Moving Average for the Series s
        Note: we blithely assume s is ordered, contiguous and without missing data"""

    # - some simple sanity checks
    if not isinstance(s, pd.core.series.Series):
        raise TypeError('The s argument should be a pandas Series')
    if not isinstance(n, int):
        raise TypeError('The n argument must be an integer')
    if n < 5:
        raise ValueError('The n argument must be >= 5')
    if n % 2 == 0:
        raise ValueError('The n argument must be odd')
    if len(s) < n:
        raise ValueError('The s argument should be a Series longer than n')
    if(((~np.isfinite(s)) | np.isnan(s)).any()):
        raise ValueError('The s argument should be finite and not_null in all cases')


    # - calculate the symmetric weights
    w = hmaSymmetricWeights(n)

    # preliminaries
    r = pd.Series(np.repeat(np.nan, len(s)), index=s.index) # the empty return vehicle
    m = int((n-1)//2)
    l = len(s)

    # - and now move over the length of the series ...
    for i in range(len(s)) :
        if i < m:
            # --- head section of series
            awlen = m+i+1
            u = hmaAsymmetricWeights(awlen, w)[::-1] # reverse - asymmetric to the left
            #print(s.iloc[0:awlen])
            r.iloc[i] = (s.iloc[0:awlen] * u).sum()
        elif i + m >= l:
            # --- tail section of series
            u = hmaAsymmetricWeights(m+l-i, w)
            r.iloc[i] = (s.iloc[(i-m):l] * u).sum()
        else:
            # --- middle section of series
            r.iloc[i] = (s.iloc[(i-m):(i+m+1)] * w).sum()
    #print(r[1])
    return (r)


### - test code
#--------------
# Check against Table 1 in B Quenneville and B Lefrancois (2001),
# "Implicit Forecasts in Musgrave Asymmetric Averages",
# Proceedings of the Annual Meeting of the American Statistical Association,
# August 5-9, 2001.
#--------------
#w = hmaSymmetricWeights(9)
#print(w)
#print(w.sum()) # should be one
#u = hmaAsymmetricWeights(7, w)
#print(u)
#print(u.sum()) # should be one
#--------------
#print (Henderson(pd.Series(range(30))+pd.Series(np.random.randn(30)), 13))