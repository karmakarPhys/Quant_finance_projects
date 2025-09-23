**Markowitz.py: **

This script demonstrates the advantages of a diversified portfolio with shares of various stocks. One can change the stocks they want to include in the portfolio 
by modifying the line: stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']. It downloads the dataset from YahooFinance the start and end date of which can be
modified in the main part of the script, along with the number of diversified portfolios (Nportfolios). It creates Nportfolios numbers of portfolios with 
different weights (share) of the stocks provided by the user. It calculates the expected returns and volatilities of the portfolios from the log returns of individual stocks. The function sharpe_ratio can be used to find the Sharpe ratio of each portfolio. The following functions are used to obtain and visualize various special portfolios:
1. min_vol_portfolio: finds the portfolio with minimum expected volatility in terms of weights of individual stocks.  
2. efficient_risk_portfolio: for a fixed expected volatility, finds the weights corresponding to the portfolio with maximum expected returns.
3. efficient_return_portfolio: for a fixed expected return value, finds the weights corresponding to the portfolio with minimum expected volatility.
4. show_optimal_portfolio: shows the portfolio with highest Sharpe ratio.
5. show_CML: plots the capital market line.
6. show_all_portfolio: plots all interest portfolios the user wishes to visualize (takes boolean inputs). 

**bs-pde2.ipynb**

Solves the Black-Scholes equation for a European call option on a single underlier, using explicit finite difference method. 
