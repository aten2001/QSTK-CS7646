import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep
import datetime as dt
import matplotlib.pyplot as plt
import pandas
from pylab import *
from matplotlib import gridspec

#to generate plot of rolling averages, can use most of code from https://github.com/tucker777/QSTK/blob/master/Examples/Basic/movingavg-ex.py
#
# Prepare to read the data
#
symbols = ["MSFT"]
startday = dt.datetime(2010,1,1)
endday = dt.datetime(2010,12,31)
timeofday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

dataobj = da.DataAccess('Yahoo')
voldata = dataobj.get_data(timestamps, symbols, "volume")
adjcloses = dataobj.get_data(timestamps, symbols, "close")
actualclose = dataobj.get_data(timestamps, symbols, "actual_close")

#adjcloses = adjcloses.fillna()
adjcloses = adjcloses.fillna(method='backfill')
adjcloses=adjcloses[symbols]

rolling_means = pandas.rolling_mean(adjcloses,20,min_periods=20)
rolling_stds = pandas.rolling_std(adjcloses,20,min_periods=20)
upperband = rolling_means + rolling_stds
lowerband = rolling_means - rolling_stds
Bollinger_val = (adjcloses - rolling_means) / (rolling_stds)

# Plot the prices
plt.clf()
#symtoplot = 'AAPL'
fig=plt.figure()
gs=gridspec.GridSpec(2,1)
ax1=fig.add_subplot(gs[0,:])
ax1.plot(adjcloses.index,adjcloses[symbols].values,label=symbols)
ax1.plot(adjcloses.index,rolling_means[symbols].values)
#upper band
ax1.plot(adjcloses.index,upperband[symbols].values)
#lower band
ax1.plot(adjcloses.index,lowerband[symbols].values)
ax1.legend([symbols,'Moving Avg.', 'Upper band', 'Lower band'], loc='best', prop={'size':8})
ax1.set_ylabel('Adjusted Close')
locs, labels = plt.xticks(size='x-small')
plt.yticks(size='x-small')
plt.setp(labels,rotation=15)

for i in range(150):
	print adjcloses.index[i:i+1]
	print Bollinger_val[symbols].values[i:i+1]

ax2=fig.add_subplot(gs[1,:])
ax2.plot(adjcloses.index,Bollinger_val[symbols].values)
#ax2.legend(['Bollinger Value'], loc='best', prop={'size':10})
ax2.set_ylabel('Bollinger Value')

locs, labels = plt.xticks(size='x-small')
plt.yticks(size='x-small')
plt.setp(labels,rotation=15)

savefig("hw5_plot.pdf", format='pdf')
