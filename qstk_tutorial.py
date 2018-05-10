import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ls_symbols = ["AAPL", "GLD", "GOOG", "$SPX", "XOM"]
dt_start = dt.datetime(2006, 1, 1)
dt_end = dt.datetime(2010, 12, 31)
dt_timeofday = dt.timedelta(hours=16)
# The reason we need to specify 16:00 hours is because we want to read the data
#that was available to us at the close of the day.
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
#above two lines provide the various data types you want to read.
ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
#creates a list of dataframe objects which have all the different types of data.
d_data = dict(zip(ls_keys, ldf_data))
# converts this list into a dictionary and then we can access anytype of data we want easily.

na_price = d_data['close'].values

plt.clf()
plt.plot(ldt_timestamps, na_price)
plt.legend(ls_symbols)
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
plt.savefig('adjustedclose.pdf', format='pdf')

#to get relative, or normalized, price of equities wrt other equities
na_normalized_price = na_price / na_price[0, :]
plt.clf()
plt.plot(ldt_timestamps, na_normalized_price)
plt.legend(ls_symbols)
plt.ylabel('Normalized Close')
plt.xlabel('Date')
plt.savefig('normalizedclose.pdf', format='pdf')

#retRun Python fileurns by day
na_rets = na_normalized_price.copy()
tsu.returnize0(na_rets)
#returnize0 calculates the daily returns of the prices
plt.clf()
plt.plot(ldt_timestamps[0:50],na_rets[0:50,3]) # $SPX 50 days
plt.plot(ldt_timestamps[0:50],na_rets[0:50,4]) # XOM 50 days
plt.axhline(y=0,color='r')
plt.legend(['$SPX','XOM'])
plt.ylabel('Daily Returns')
plt.xlabel('Date')
plt.savefig('dailyreturns.pdf',format='pdf')

def simulate(dt_start, dt_end, ls_symbols, ls_allocations):
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    na_price = d_data['close'].values
    # see above for meaning of these lines

    na_normalized_price = na_price / na_price[0, :]
    #na_price[0,:] returns 1st row of na_price,
    #which are values for 1st day for each ls_symbols

    #if ls_symbols=['AAPL', 'GLD', 'GOOG', 'XOM' then
    #na_price[0:50,3]) price for GOOG for 50 days
    #na_price[0:50,4]) price for XOM for 50 days

    #ls_allocations is input parameter
    # Allocations to the equities at the beginning of the simulation (e.g., 0.2, 0.3, 0.4, 0.1)

    #row-wise multiplication by weights
    na_weighted_price = na_normalized_price * ls_allocations
	#na_normalized_price[rows are days, columns are stocks]
	#ex: na_normalized_price[13,1] is normalized price on 14th day for 'GLD'
	
    #row-wise sum
    total_weightprice_perday = na_weighted_price.copy().sum(axis=1);
    #total_weightprice_perday is a Nx1 vector, where N is #of days (say, 50)
	#because it takes the sum of every stock per day
	#ex: na_weighted_price=[ ['AAPL', 'GLD', 'GOOG', 'XOM'],
	#					(1st day) [.35, .3, .5, .2],
	#						  ...,
	#					(last day)	  [.5,.3,.6,.7] ]
	#  total_weightprice_perday = [1.35 (1st day),
	#								...,
	#					(last day)	2.1]
	
	#np.sum([[2, 3], [4, 5]], axis=1) gives array([5,9])
	#np.sum([[2, 3], [4, 5]], axis=0) gives array([6,8])
	

    #Calculate daily returns on portfolio
    rets_weightprice_perday = total_weightprice_perday.copy()
    tsu.returnize0(rets_weightprice_perday)

    vol = np.std(rets_weightprice_perday)
    #vol = np.std(na_weighted_rets)
    #get standard dev of weighted returns

    daily_ret = np.average(rets_weightprice_perday)
    #daily_ret = np.average(na_weighted_rets)

    #Sharpe ratio = (Mean portfolio return - Risk-free rate)/Standard deviation of portfolio return
    #problem states "Always assume you have 252 trading days in an year. And risk free rate = 0"
    sharpe = np.sqrt(252) * daily_ret / vol

    cum_ret = np.dot(na_price[-1]/na_price[0], np.transpose(ls_allocations))
    #negative index for array accesses elements starting with last index
    # ex) a= [1, 2, 3]
    #>>> print a[-2]
    #2
    #>>> print a[-1]
    #3
    #earlier, we had na_normalized_price = na_price / na_price[0, :]

    #print na_price[0, :]
    #[   74.43    53.12   435.23  1268.8     50.47]
    #print na_price[0]
    #[   74.43    53.12   435.23  1268.8     50.47]
    #print na_price[-1]
    #[  322.28   137.03   598.86  1257.88    70.33]

    return vol, daily_ret, sharpe, cum_ret

startdate = dt.datetime(2011, 1, 1)
enddate = dt.datetime(2011, 12, 31)
vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ['AAPL', 'GLD', 'GOOG', 'XOM'], [0.4, 0.4, 0.0, 0.2])
print "Sharpe Ratio:", sharpe
print "Volatility (stdev of daily returns): ", vol
print "Average Daily Return:", daily_ret
print "Cumulative Return:", cum_ret

startdate = dt.datetime(2010, 1, 1)
enddate = dt.datetime(2010, 12, 31)
vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ['AXP', 'HPQ', 'IBM', 'HNZ'], [0.0, 0.0, 0.0, 1.0])
print "Sharpe Ratio:", sharpe
print "Volatility (stdev of daily returns): ", vol
print "Average Daily Return:", daily_ret
print "Cumulative Return:", cum_ret