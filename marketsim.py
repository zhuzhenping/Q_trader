"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    order_book = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    order_book = order_book.sort_index()
    start_date = order_book.index[0]
    end_date = order_book.index[-1]
    Symbols = []
    for i in range(order_book.shape[0]):
        i_symbol = order_book['Symbol'].iat[i]
        if i_symbol not in Symbols: Symbols.append(i_symbol)
    
    Price = get_data(Symbols, pd.date_range(start_date, end_date))
    Price = Price[Symbols]  # remove SPY
    Price['Cash'] = np.ones(Price.shape[0], dtype='float')
    Labels = Symbols + ['Cash']
    
    Trade = Price.copy()
    Trade[Labels] = np.zeros(Trade.shape, dtype='float')
    Hold = Trade.copy()
    Hold[Labels] = np.zeros(Hold.shape, dtype='float')
    Value = Trade.copy() 

    secret = pd.Timestamp('20110615')
    action_i = 0
    date = order_book.index[action_i]
    for i in range(Price.shape[0]):
        while Trade.index[i] == date:
            if Trade.index[i] != secret:
                sym, move, share = order_book.iloc[action_i]
                ###leverage###
                Trade_current = Trade.loc[date].copy()
                if move == 'BUY':
                    Trade_current[sym] = share
                    Trade_current['Cash'] = -Price.loc[date, sym] * share
                elif move == 'SELL':
                    Trade_current[sym] = -share
                    Trade_current['Cash'] = Price.loc[date, sym] * share
                Hold_current = Hold.loc[date].copy()
                if i == 0:
                    Hold_current['Cash'] = start_val
                    Hold_current += Trade.iloc[0]
                else:
                    Hold_current = Hold.iloc[i-1] + Trade_current
                Value_current = Hold_current * Price.iloc[i]   
                leverage = np.abs(Value_current[Symbols].values).sum() / (Value_current[Symbols].values.sum() + Value_current['Cash'])   
                Trade.loc[date] = Trade_current
            action_i += 1
            if action_i < order_book.shape[0]:
                date = order_book.index[action_i]
            else: break
        
        if i == 0:
            Hold.iloc[0]['Cash'] = start_val
            Hold.iloc[0] += Trade.iloc[0]
        else:
            Hold.iloc[i] = Hold.iloc[i-1] + Trade.iloc[i]
        Value.iloc[i] = Hold.iloc[i] * Price.iloc[i]
    
    portvals = Value.sum(axis=1)

    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

def author():
    return 'yjiao43' # Georgia Tech ID for mailbox

if __name__ == "__main__":
    test_code()
