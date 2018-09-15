# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 00:19:33 2018

@author: Admin
"""

import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimization

stocks = ['AAPL','WMT','TSLA','GE','AMZN','DB']

start_date = '01/01/2010'
end_date = '01/01/2017'

def download_data(stocks):
    data = web.DataReader(stocks,data_source="yahoo",start=start_date, end=end_date)['Adj Close']    
    data.columns = stocks
    return data

download_data(stocks)
    
def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()
show_data(download_data(stocks))
   
#natural logarithm for normalization process
def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns

def plot_daily_returns(returns):
    returns.plot(figsize=(10,5))
    plt.show()
    
def show_statistics(returns):
    print(returns.mean()*252) # 252 is the no of trading days
    print(returns.cov()*252)


def initialize_weights():
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights

def calculate_portfolio_return(returns, weights):
    portfolio_return = np.sum(returns.mean()*weights)*252
    print('Expected portfolio return : ', portfolio_return )

def calculate_portfolio_variance(returns, weights):
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights)))
    print('Expected variance : ', portfolio_variance )

def generate_portfolios(weights, returns):
    preturns = []
    pvariances = []
    for i in range(1000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvariances.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights))))
    preturns = np.array(preturns)
    pvariances = np.array(pvariances)
    return preturns, pvariances

def plot_portfolios(returns, variances):
    plt.plot(figsize=(10,5))
    plt.scatter(variances, returns, c= returns/variances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar('Sharpe Ratio')
    plt.show()
    
def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean()*weights)*252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

def min_func_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints =({'type':'eq','fun':lambda x:np.sum(x)-1})
    bounds = tuple((0,1) for x in range(len(stocks)))
    optimum = optimization.minimize(fun=min_func_sharpe, x0=weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimum

def print_optimal_portfolio(optimum, returns):
    print('Optimal weights', optimum['x'].round(3))
    print('Expected return, volatility and Sharpe Ratio:', statistics(optimum['x'].round(3), returns))
    
def show_optimal_portfolio(optimum, returns, preturns, pvariances):
    plt.figure(figsize=(10,6))
    plt.scatter(pvariances, preturns, c= preturns/pvariances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar('Sharpe Ratio')
    plt.plot(statistics(optimum['x'],returns)[1], statistics(optimum['x'],returns)[0],'g*', markersize=20.0)
    plt.show()
    
if __name__ == '__main__':
    data = download_data(stocks)
    show_data(data)
    returns = calculate_returns(data)
    plot_daily_returns(returns)
    show_statistics(returns)
    weights=initialize_weights()
    calculate_portfolio_return(returns,weights)
    calculate_portfolio_variance(returns,weights)
    preturns,pvariances=generate_portfolios(weights, returns)
    plot_portfolios(preturns,pvariances)
    optimum=optimize_portfolio(weights,returns)
    print_optimal_portfolio(optimum, returns)
    show_optimal_portfolio(optimum, returns, preturns, pvariances)