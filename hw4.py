import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import csv

def marketsim(starting_cash, order_file,out_file):
    dates = []
    symbols = []
    #order_list=[]

    #starting_cash = float(sys.argv[1])
    #order_file = sys.argv[2]
    #out_file = sys.argv[3]

    #step1: read in csv file and remove duplicates
    #see marketsim-guidelines.pdf
    reader = csv.reader(open(order_file, 'rU'), delimiter=',')
    for row in reader:
        #ex: 2008, 12, 3, AAPL, BUY, 130
        dates.append(dt.datetime(int(row[0]), int(row[1]), int(row[2])))
	    #need int, otherwise get "TypeError: an integer is required"
        symbols.append(row[3])

    #order_list.sort(['date'])

    #remove duplicates
    #set(listWithDuplicates) is an unordered collection without duplicates
    #so it removes the duplicates in listWithDuplicates
    uniqueDates = list(set(dates))
    uniqueSymbols = list(set(symbols))


    #step 2 - read the data like in previous HW and tutorials
    sortedDates=sorted(uniqueDates)
    dt_start = sortedDates[0]
    #End date should be offset-ed by 1 day to
    #read the close for the last date. - see marketsim-guidelines.pdf
    dt_end = sortedDates[-1] + dt.timedelta(days=1)

    dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    ldf_data = dataobj.get_data(ldt_timestamps, uniqueSymbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    #step 3: create dataframe that contains trade matrix
    #see marketsim-guidelines.pdf
    df_trade = np.zeros((len(ldt_timestamps), len(uniqueSymbols)))
    df_trade = pd.DataFrame(df_trade, index=[ldt_timestamps], columns=uniqueSymbols)

    #iterate orders file and fill the number of shares for that 
    #symbol and date to create trade matrix


    reader = csv.reader(open(order_file, 'rU'), delimiter=',')
    for orderrow in reader:
        order_date = dt.datetime(int(orderrow[0]), int(orderrow[1]), int(orderrow[2])).date()
        for index, row in df_trade.iterrows():
            if order_date == index.date():
                if orderrow[4]=='Buy':
		            df_trade.set_value(index, orderrow[3], float(orderrow[5]))
		            #df_trade.ix[index][orderrow[3]] += float(orderrow[5])
		            #print ts_cash[index]
                elif orderrow[4]=="Sell":
		            #df_trade.ix[index][orderrow[3]] -= float(orderrow[5])
		            df_trade.set_value(index, orderrow[3], -float(orderrow[5]))
    print df_trade	   

    #step4: create timeseries containing cash values, all values are 0 initially
    ts_cash = pd.TimeSeries(0.0, index=ldt_timestamps)
    ts_cash[0] = starting_cash
    #for each order in trade matrix, subtract the cash used in that trade
    for index,row in df_trade.iterrows():
       ts_cash[index] -= np.dot(row.values.astype(float), d_data['close'].ix[index])


    #print 'df_trade',df_trade.head()
    #step5: 
    #append '_CASH' into the price date	   
    df_close = d_data['close']		   
    df_close['_CASH']=1.0	

    #append cash time series into the trade matrix
    df_trade['_CASH'] = ts_cash

    #convert to holding matrix
    df_holding = df_trade.cumsum()
    #df_trade = df_trade.cumsum(axis=1)
    #axis=1 means sum over columns
    #see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.cumsum.html

    #dot product on price (df_close) and holding/trade matrix (df_trade) to 
    #calculate portfolio on each date
    ts_fund = np.zeros((len(ldt_timestamps), 1))
    #ts_fund = pd.DataFrame(ts_fund, index=ldt_timestamps, columns='portfolio value')

    ts_fund = df_holding.mul(df_close, axis='columns', fill_value=0).sum(axis =1)
    #better to avoid iterating over rows unless necessary
    #and try to use pandas' vectorized operations
    #for index, row in df_trade.iterrows():
    #        portfolio_value = np.dot(row.values.astype(float), df_close.ix[index].values)
    #        ts_fund[index] = portfolio_value
		
    #write this to csv
    writer = csv.writer(open(out_file, 'wb'), delimiter=',')
    for row_index in ts_fund.index:
        row_to_enter = [row_index.year, row_index.month, row_index.day, ts_fund[row_index]]
        writer.writerow(row_to_enter)
		
    return out_file

	
def analyze(portfolio_file, benchmark_symbol):
    dates = []
    portfolio_values = []
	
    #portfolio_file = sys.argv[1]
    #benchmark_symbol = sys.argv[2]

	#read in values.csv
    reader = csv.reader(open(portfolio_file, 'rU'), delimiter=',')
    for order in reader:
        dates.append(dt.datetime(int(order[0]), int(order[1]), int(order[2])))
        portfolio_values.append(float(order[3]))
 
    dt_start = dates[0]
    #End date should be offset-ed by 1 day to
    #read the close for the last date. - see marketsim-guidelines.pdf
    dt_end = dates[-1] + dt.timedelta(days=1)
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(ldt_timestamps, benchmark_symbol, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
	
	#remove NAN from price data
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
		
    na_price_benchmark = d_data['close'].values
    # see above for meaning of these lines
	
	#convert list to array
    portfolio_values=np.array(portfolio_values)

    na_normalized_price_portfolio = portfolio_values
    na_normalized_price_benchmark = na_price_benchmark / na_price_benchmark[0, :]
    
    #if ls_symbols=['AAPL', 'GLD', 'GOOG', 'XOM' then
    #na_price[0:50,3]) price for GOOG for 50 days
    #na_price[0:50,4]) price for XOM for 50 days
	
    #Calculate avg daily returns on benchmark and portfolio
    na_normalized_price_portfolio_perday = na_normalized_price_portfolio.copy()
    tsu.returnize0(na_normalized_price_portfolio_perday)
	
    na_normalized_price_benchmark_perday = na_normalized_price_benchmark.copy()
    tsu.returnize0(na_normalized_price_benchmark_perday)

    daily_ret_portfolio = np.average(na_normalized_price_portfolio_perday)
    daily_ret_benchmark = np.average(na_normalized_price_benchmark_perday)
    #daily_ret = np.average(na_weighted_rets)
	
    #get standard dev of weighted returns
    vol_portfolio = np.std(na_normalized_price_portfolio_perday)
    vol_benchmark = np.std(na_normalized_price_benchmark_perday)
    #vol = np.std(na_weighted_rets)

    #Sharpe ratio = (Mean portfolio return - Risk-free rate)/Standard deviation of portfolio return
    #problem states "Always assume you have 252 trading days in an year. And risk free rate = 0"
    sharpe_portfolio = np.sqrt(252) * daily_ret_portfolio / vol_portfolio
    sharpe_benchmark = np.sqrt(252) * daily_ret_benchmark / vol_benchmark

	#cumulative return of total portfolio
    cum_ret_portfolio = portfolio_values[-1]/portfolio_values[0]
    cum_ret_benchmark = na_price_benchmark[-1]/na_price_benchmark[0]
    
    mult = portfolio_values[0]/na_price_benchmark[0]
	
    plt.clf()
    plt.plot(ldt_timestamps, portfolio_values)
    plt.plot(ldt_timestamps, na_price_benchmark*mult)
    plt.legend(['Portfolio', 'Benchmark'], loc='best')
    plt.ylabel('Fund Value', size='x-small')
    plt.xlabel('Date', size='x-small')
    locs, labels = plt.xticks(size='x-small')
    plt.yticks(size='x-small')
    plt.setp(labels,rotation=15)
    #for hw3:
    #plt.savefig('HW3_CumPortfolioValues.pdf', format='pdf')
	#for hw4:
    plt.savefig('HW4_CumPortfolioValues.pdf', format='pdf')
	
    print "Details of the Performance of the portfolio :"
    print "Data Range :", str(dates[0]), " to ",str(dates[-1])
    print "Sharpe Ratio of Fund:", sharpe_portfolio
    print "Sharpe Ratio of $SPX:", sharpe_benchmark
    print "Standard Deviation of Fund: ", vol_portfolio
    print "Standard Deviation of $SPX: ", vol_benchmark
    print "Average Daily Return of Fund:", daily_ret_portfolio
    print "Average Daily Return of $SPX:", daily_ret_benchmark
    print "Total Return of Fund:", cum_ret_portfolio
    print "Total Return of $SPX:", cum_ret_benchmark
	
    #return vol_portfolio, daily_ret_portfolio, sharpe_portfolio, cum_ret_portfolio, 
    #   vol_benchmark, daily_ret_benchmark, sharpe_benchmark, cum_ret_benchmark




def find_events(ls_symbols, d_data, output_file):
    #create an event matrix we start by reading the data for the specified 
    #time duration as mentioned in the tutorial 1. Then we calculate 
    #normalized returns for the equity data. 
    
    #c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)

    #remove NAN from price data
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
		
    df_close = d_data['actual_close']
		
    trade_events=[]
    buysell=[]
	
    #timestamps for event range
    ldt_timestamps=df_close.index 
	
    #market price, which is indicated by 'SPY'
    ts_market = df_close['SPY']
	
    writer = csv.writer(open(output_file, 'wb'), delimiter=',')	
    for s_sym in ls_symbols: # for each symbol
        for i in range(1, len(ldt_timestamps)): # for each day
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

            # Event is found if price[t-1] >= 5.0 price[t] < 5.0 
			# Instead of putting a 1 in the event matrix, output to a file:
			#ex: 
			#Date, AAPL, BUY, 100
			#Date + 5 days, AAPL, SELL, 100
            if f_symprice_yest >= 5.0 and f_symprice_today < 5.0:
                #instead of df_events[s_sym].ix[ldt_timestamps[i]] = 1,
                #When an event occurs, buy 100 shares of the equity on that day.
                #Sell automatically 5 trading days later. 
                #trade_events.append([ldt_timestamps[i].year, ldt_timestamps[i].month, 
				#    ldt_timestamps[i].day, s_sym, 'Buy', 100])
                row_to_enter =  [ldt_timestamps[i].year, ldt_timestamps[i].month, ldt_timestamps[i].day, s_sym, 'Buy', 100]
                writer.writerow(row_to_enter)	
                try:
                    #trade_events.append([ldt_timestamps[i+5].year, ldt_timestamps[i+5].month, ldt_timestamps[i+5].day, s_sym, 'Sell', 100])
                    #break
                    time_n = ldt_timestamps[i + 5]
                    row_to_enter =  [ldt_timestamps[i + 5].year, ldt_timestamps[i + 5].month, ldt_timestamps[i + 5].day, s_sym, 'Sell', 100]
                    writer.writerow(row_to_enter)					
                except IndexError:
                    print('For the final few events there is less than 5 days left')
                    #assume that you exit on the last day, so hold it less than 5 days. 
                    #trade_events.append([ldt_timestamps[-1].year, ldt_timestamps[-1].month, ldt_timestamps[-1].day, s_sym, 'Sell', 100])
                    time_n = ldt_timestamps[-1]
                    row_to_enter =  [ldt_timestamps[-1].year, ldt_timestamps[-1].month, ldt_timestamps[-1].day, s_sym, 'Sell', 100]	
                    writer.writerow(row_to_enter)
    #write this to csv
    
    #for row_to_enter in trade_events:
    #    writer.writerow(row_to_enter)			
    
    return trade_events

	
dt_start = dt.datetime(2008, 1, 1)
dt_end = dt.datetime(2009, 12, 31)
dt_timeofday = dt.timedelta(hours=16)
#dt_start=dt.datetime(2008, 1, 1)
#dt_end=dt.datetime(2009, 12, 31)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

#read list of stocks in S&P 500 in 2008 using the QSTK call 
dataobj = da.DataAccess('Yahoo')
ls_symbols = dataobj.get_symbols_from_list("sp5002012")
ls_symbols.append('SPY')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)

d_data = dict(zip(ls_keys, ldf_data))
 
df_events = find_events(ls_symbols, d_data, 'hw4_orders.csv')
#print "Creating Study"
#ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
#                s_filename='HW2_EventStudy_2012.pdf', b_market_neutral=True, b_errorbars=True,
#                s_market_sym='SPY')


#after this program finishes, to see how your Event Study (series of trades based on Events)
#run "hw3_marketsim.py 50000 hw4_orders.csv hw4_values.csv" to execute the trades in the Event Study
#then run "hw3_analyze.py hw4_values.csv SPX" to assess the performance of the Event Study
#and see how it compares to the benchmark

outputf = marketsim(50000, 'hw4_orders.csv', 'hw4_values.csv')
#for hw4:
vol_p, daily_p, sharpe_p, cum_p, vol_b, daily_b, sharpe_b, cum_b = analyze(outputf, ['$SPX'])
