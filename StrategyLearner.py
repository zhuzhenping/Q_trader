"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np
import matplotlib.pyplot as plt

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = None
        self.SMA_Nday = 25
        self.num_of_bin = 10
        
    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)[[symbol]]
        daily_returns = prices.copy()
        daily_returns[1:] = (prices[1:]/prices[:-1].values)-1
        daily_returns.ix[0] = 0
        
        SMA_price = self.SMA(sd, ed, symbol, self.SMA_Nday)
        bb = self.Bollinger(sd, ed, symbol, self.SMA_Nday)
        RSI_value = self.RSI(sd, ed, symbol, self.SMA_Nday)
        
        dis_SMA = self.discretize(SMA_price,  self.num_of_bin)
        dis_bb = self.discretize(bb,  self.num_of_bin)
        dis_RSI = self.discretize(RSI_value,  self.num_of_bin)
        State_list = dis_SMA *  self.num_of_bin**2 + dis_bb *  self.num_of_bin + dis_RSI
        self.learner = ql.QLearner(num_states=np.int( self.num_of_bin**3),\
        num_actions = 3, \
        alpha = 0.4, \
        gamma = 0.9, \
        rar = 0.99, \
        radr = 0.9999, \
        dyna = 0, \
        verbose=False) #initialize the learner
        # 3 actions: 0--OUT; 1--LONG; 2--SHORT
        iteration = 0
        prev_r = -10
        while iteration < 100:
            x = State_list.iloc[0]
            self.learner.querysetstate(x)
            curr_hold = 0
            total_r = 0
            for i in range(1, prices.shape[0]):
                r = daily_returns.iloc[i][symbol] * curr_hold *10000
                if curr_hold == 0: r = 0
                total_r += r
                state = State_list.iloc[i]
                action = self.learner.query(state,r)
  #              print (state,action)
                if action == 0:
                    curr_hold = 0
                elif action == 1:
                    curr_hold = 1
                elif action == 2:
                    curr_hold = -1
            if (iteration > 30) and (prev_r*0.98<total_r <prev_r*1.02):
                break
         #   print total_r
            prev_r = total_r
            iteration += 1

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms] # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices
  
        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 10000):
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = ut.get_data([symbol], dates)[[symbol]]
        trades = prices.copy()  
        trades.values[:] = 0 # set them all to nothing
        
        SMA_price = self.SMA(sd, ed, symbol, self.SMA_Nday)
        bb = self.Bollinger(sd, ed, symbol, self.SMA_Nday)
        RSI_value = self.RSI(sd, ed, symbol, self.SMA_Nday)
        
        dis_SMA = self.discretize(SMA_price,  self.num_of_bin)
        dis_bb = self.discretize(bb,  self.num_of_bin)
        dis_RSI = self.discretize(RSI_value,  self.num_of_bin)
        State_list = dis_SMA *  self.num_of_bin**2 + dis_bb *  self.num_of_bin + dis_RSI
        curr_hold = 0
        for i in range(trades.shape[0]):
            if i == 0:
                r = 0
            else:
                r = (prices.iloc[i]/prices.iloc[i-1] - 1) * curr_hold*10000
                if curr_hold == 0: r = -400
            state = State_list.iloc[i]
            action = self.learner.query(state,r)
            if action == 0:
                if curr_hold == 1: trades.iloc[i] = -200
                if curr_hold == -1: trades.iloc[i] = 200
                curr_hold = 0 
            if action == 1:
                if curr_hold == 0: trades.iloc[i] = 200
                if curr_hold == -1: trades.iloc[i] = 400
                curr_hold = 1
            elif action == 2:
                if curr_hold == 1: trades.iloc[i] = -400
                if curr_hold == 0: trades.iloc[i] = -200
                curr_hold = -1
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades 
        
    def BenchmarkPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 200 # add a BUY at the 4th date
        trades.values[-1,:] = -200 # add a SELL at the 6th date 
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades
              
    def SMA(self,sd, ed, stock, SMA_Nday):
        dates = pd.date_range(sd, ed)
        prices1 =ut.get_data([stock], dates)
        nsd = sd - dt.timedelta(days = 100)
        ndates = pd.date_range(nsd, ed)
        prices = ut.get_data([stock], ndates)
        prices_apple = prices[[stock]]
        # plot_data(prices)
        SMA_apple = pd.rolling_mean(prices_apple,SMA_Nday,min_periods=1,center=False)
        price_SMA = prices_apple/SMA_apple
        price_SMA_norm = (price_SMA - price_SMA.mean())/price_SMA.std()
    #    price_SMA = price_SMA/price_SMA.ix[SMA_Nday]
        if False:
            SMA_apple = SMA_apple/prices_apple.ix[0]
            prices_n = prices_apple/prices_apple.ix[0]
            SMA_all = pd.concat([prices_n, SMA_apple, price_SMA], keys=['Price', 'SMA', 'Price/SMA'], axis=1)
            ut.plot_data(SMA_all, title='Technical indicator: SMA', ylabel='Relative Value')
        return pd.DataFrame(price_SMA_norm, index=prices1.index).fillna(0)
        
    def Bollinger(self,sd, ed, stock, SMA_Nday):
        dates = pd.date_range(sd, ed)
        prices1 =ut.get_data([stock], dates)
        nsd = sd - dt.timedelta(days = 100)
        ndates = pd.date_range(nsd, ed)
        prices = ut.get_data([stock], ndates)
        prices_apple = prices[[stock]]
        # plot_data(prices)
        SMA_apple = pd.rolling_mean(prices_apple,SMA_Nday,min_periods=1,center=False)
        STD_apple = pd.rolling_std(prices_apple,SMA_Nday,min_periods=1,center=False)
        Bollinger_up = SMA_apple + 2*STD_apple
        Bollinger_down = SMA_apple - 2*STD_apple
        bb_percent = (prices_apple - Bollinger_down)/(Bollinger_up - Bollinger_down)
        bb_percent_norm = (bb_percent - bb_percent.mean())/bb_percent.std()
        if False:
            SMA_apple = SMA_apple/prices_apple.ix[0]
            Bollinger_up = Bollinger_up/prices_apple.ix[0]
            Bollinger_down = Bollinger_down/prices_apple.ix[0]
            prices_1 = prices_apple/prices_apple.ix[0]
            bb_all = pd.concat([prices_1, bb_percent], keys=['Price', 'bb%'], axis=1)
            bb_all.plot(title='Technical indicator: Bollinger Bands Precentage')
            plt.figure(2)
            ut.plot_data(bb_percent, title='Technical indicator: Bollinger Bands Precentage', ylabel='BB%')
        return pd.DataFrame(bb_percent_norm, index=prices1.index).fillna(0)

    def RSI(self,sd, ed, stock, SMA_Nday):
        dates = pd.date_range(sd, ed)
        prices1 = ut.get_data([stock], dates)
        nsd = sd - dt.timedelta(days = 100)
        ndates = pd.date_range(nsd, ed)
        prices = ut.get_data([stock], ndates)
        prices_apple = prices[[stock]]
        price_delta = prices_apple.diff()
        dUp = price_delta.copy()
        dDown = price_delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        rUp = pd.rolling_mean(dDown,SMA_Nday,min_periods=1,center=False).abs()
        rDown = pd.rolling_mean(dUp,SMA_Nday,min_periods=1,center=False).abs()
        RS = rUp/rDown
        RSI = 100 - 100/(1+RS)
        RSI_norm = (RSI - RSI.mean())/RSI.std()
        if False:
            prices_1 = prices_apple/prices_apple.ix[0]
            RSI_1 = RSI/RSI.ix[SMA_Nday]
            RSI_all = pd.concat([prices_1, RSI_1], keys=['Price', 'RSI'], axis=1)
            plt.figure(3)
            ut.plot_data(RSI, title='Technical indicator: Relative Strength Index', ylabel='RSI')
        return pd.DataFrame(RSI_norm, index=prices1.index).fillna(0)
    def discretize(self, indicator, num_states):
        label = range(num_states-1)
        res = pd.qcut(indicator[indicator.columns[0]] ,num_states-1, labels = label)
        res = res.cat.codes
        return res.to_frame().astype(float)

if __name__=="__main__":
    print "One does not simply think up a strategy"
