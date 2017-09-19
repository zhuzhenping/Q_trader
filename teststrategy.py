"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""
import os
os.chdir('C:/Users/yzhu/Google Drive/Yuge/Gatech/ML4T/ML4T_2017Spring/mc3p4_qlearning_trader/')
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import util as ut
import marketsim as ms
import StrategyLearner as sl
import numpy as np
def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "AAPL"
    stdate =dt.datetime(2008,1,1)
    enddate =dt.datetime(2009,12,31) # just a few days for "shake out"
    starting_value = 100000
    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = starting_value) 
#    df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv=starting_value)
#    bench_trades = learner.BenchmarkPolicy(symbol = sym, sd = stdate, ed = enddate, sv=starting_value)
 #   df_port = compute_portvals_df(df_trades, starting_value)
 #   bench_port =  compute_portvals_df(bench_trades, starting_value)
    
    # set parameters for testing
 #   sym = "AAPL"
 #   stdate =dt.datetime(2010,1,1)
 #   enddate =dt.datetime(2011,12,31)
    df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv=starting_value)
    bench_trades = learner.BenchmarkPolicy(symbol = sym, sd = stdate, ed = enddate, sv=starting_value)
    df_port = compute_portvals_df(df_trades, starting_value)
    bench_port =  compute_portvals_df(bench_trades, starting_value)
    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=200] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 200 shares)"
    if verb: print df_trades
    
    
    portvals_all = pd.concat([bench_port, df_port], keys=['Benchmark', 'Q-learning'], axis=1)
    portvals_all /= starting_value
    ut.plot_data(portvals_all, title='Q-trader vs Benchmark', ylabel='Relative Portfolio Value')    
    
def compute_portvals_df(orders_df, sv):
    """"
    compute portfolio value from a Trade dataframe
    """
    Symbol = [orders_df.columns[0]]
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    Price = ut.get_data(Symbol, pd.date_range(start_date, end_date))[Symbol]
    Value = orders_df.copy()
    Value[Symbol] = np.zeros(Value.shape, dtype='float')
    Cash = orders_df.copy()
    Cash[Symbol] = np.zeros(Value.shape, dtype='float')
    Cash.iloc[0][Symbol] = sv
    Hold = 0
    Cash.iloc[0][Symbol] -= orders_df.iloc[0][Symbol] * Price.iloc[0][Symbol]
    Hold += orders_df.iloc[0][Symbol]
    Value.iloc[0][Symbol] = Cash.iloc[0][Symbol] + Hold * Price.iloc[0][Symbol]    
    for i in range(1,Value.shape[0]):
        Cash.iloc[i][Symbol] = Cash.iloc[i-1][Symbol] - orders_df.iloc[i][Symbol] * Price.iloc[i][Symbol]
        Hold += orders_df.iloc[i][Symbol]
        Value.iloc[i][Symbol] = Cash.iloc[i][Symbol] + Hold * Price.iloc[i][Symbol]
    return Value
    
    
if __name__=="__main__":
    test_code(verb = False)
