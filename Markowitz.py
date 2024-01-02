import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
#from scipy.optimize import optimization

NUM_TRADING_DAYS = 252


# stocks we are going to handle
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

def download_data(start_date, end_date, type_data):

    stock_data = {}
    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)[type_data]
    return pd.DataFrame(stock_data)

def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def generate_portfolio(Nportfolios, returns):
	weights, portfolios_return, portfolios_var = [],[],[]
	for i in range(Nportfolios):
		w  =  np.random.random(len(stocks))
		w/=np.sum(w)
		weights.append(w)
		portfolio_return = np.dot(w.T,  returns.mean()*252 )
		portfolios_return.append(portfolio_return)
		portfolio_var = np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)) ) * np.sqrt(252)
		portfolios_var.append(portfolio_var)
	return [np.array(weights), np.array(portfolios_return), np.array(portfolios_var)]

def statistics(weights, returns):
#    [weights, portfolios_return, portfolios_var] = portfolios
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS,weights)))
    return [portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility]

def show_portfolios(portfolios):
    [weights, portfolios_return, portfolios_var] = portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolios_var, portfolios_return, c=(portfolios_return-0.05) / portfolios_var, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def sharpe_ratio(weights, returns, risk_free):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS,weights)))
    return [portfolio_return, portfolio_volatility, (portfolio_return - risk_free) / portfolio_volatility]

#def min_function_sharpe(weights, returns):
#   return -statistics(weights, returns)[2]
def expectected_ann_vol(w, returns):
    return np.sqrt(np.dot(w.T,np.dot(returns.cov()*252, w)))

def expectected_ann_return(w, returns):
    return np.dot(w.T, returns.mean()) * 252

def min_function_sharpe(weights, returns):
    return -sharpe_ratio(weights, returns, 0.05)[2]

def exp_volatility(weights, returns):
    return sharpe_ratio(weights, returns, 0.05)[1]

def min_func_returns(weights, returns):
    return -statistics(weights, returns)[0]

def optimize_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def min_vol_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=exp_volatility, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def efficient_risk_portfolio(weights, returns, volatility_fixed):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},{'type': 'eq', 'fun': lambda x: expectected_ann_vol(x, returns) - volatility_fixed})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_func_returns, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def efficient_return_portfolio(weights, returns, return_fixed):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},{'type': 'eq', 'fun': lambda x: expectected_ann_return(x, returns) - return_fixed})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=exp_volatility, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def show_optimal_portfolio(opt, rets, portfolios):
    [weights, portfolios_return, portfolios_var] = portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolios_var, portfolios_return, c=(portfolios_return-0.05) / portfolios_var, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(sharpe_ratio(opt['x'], rets, 0.05)[1], sharpe_ratio(opt['x'], rets, 0.05)[0], 'g*', markersize=20.0)
    plt.show()

def show_CML(opt, rets, portfolios):
    [weights, portfolios_return, portfolios_var] = portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolios_var, portfolios_return, c=(portfolios_return-0.05) / portfolios_var, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(sharpe_ratio(opt['x'], rets, 0.05)[1], sharpe_ratio(opt['x'], rets, 0.05)[0], 'g*', markersize=20.0)
    point1 = (0, 0.05)
    point2 = (sharpe_ratio(opt['x'], rets, 0.05)[1], sharpe_ratio(opt['x'], rets, 0.05)[0])
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color='blue', label='Straight Line')
    plt.show()

def show_all_portfolios(rets, portfolios, show_max_sharpe,  show_CML, show_min_vol, show_efficient_risk, show_efficient_return):
    [weights, portfolios_return, portfolios_var] = portfolios
    return_val = 0.30
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolios_var, portfolios_return, c=(portfolios_return-0.05) / portfolios_var, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio', cmap='viridis')
    if show_max_sharpe==True:
        opt = optimize_portfolio(weights, log_daily_returns)
        [sharpe_return, sharpe_vol,  sharpe_val] = sharpe_ratio(opt['x'], rets, 0.05)
        plt.plot(sharpe_vol, sharpe_return, 'g*', markersize=10.0,label='Max. Sharpe Ratio')
        if show_CML==True:
                point1 = (0, 0.05)
                x_values = [point1[0], sharpe_vol]
                y_values = [point1[1], sharpe_return]
                plt.plot(x_values, y_values, color='blue', label='Capital Market Line')
    if show_min_vol==True:
        opt2 = min_vol_portfolio(weights, rets)
        [sharpe_return2, sharpe_vol2,  sharpe_val2] = sharpe_ratio(opt2['x'], rets, 0.05)
        plt.plot(sharpe_vol2, sharpe_return2, 'r*', markersize=10.0,label='Min. volatility')
    if show_efficient_risk==True:
        opt3 = efficient_risk_portfolio(weights, rets, volatility_fixed=0.25)
        [sharpe_return3, sharpe_vol3,  sharpe_val3] = sharpe_ratio(opt3['x'], rets, 0.05)
        plt.plot(sharpe_vol3, sharpe_return3, 'b*', markersize=10.0,label='Efficient risk for volatility = 0.25')
    if show_efficient_return==True:
        opt4 = efficient_return_portfolio(weights, rets, return_fixed=return_val)
        [sharpe_return4, sharpe_vol4,  sharpe_val4] = sharpe_ratio(opt4['x'], rets, 0.05)
        plt.plot(sharpe_vol4, sharpe_return4, marker='*', color='black', markersize=10.0,label='Efficient return for return = 0.30')
    plt.legend()
    plt.savefig('All_portfolio.jpeg',dpi=400)
    plt.show()

if __name__ == "__main__":
    start_date = "2017-01-01"; end_date = "2022-01-01"; type_data = 'High';
    N_portfolio = 10000
    np.random.seed(42)
    data  = download_data(start_date, end_date, type_data)
    log_daily_returns = calculate_return(data)
#    print(data.tail())
    sample_portfolios = generate_portfolio(N_portfolio, log_daily_returns)
    sample_weights = sample_portfolios[0]
#    print(sample_portfolios)
#    show_portfolios(sample_portfolios)
    one_portfolio = sample_portfolios[0][1]
#    print(one_portfolio)
#    print(statistics(one_portfolio, log_daily_returns))
#    print(sharpe_ratio(one_portfolio, log_daily_returns, 0.05))
    ## MAX SHARPE PF
    opt_portfolio = optimize_portfolio(sample_portfolios[0], log_daily_returns)
#    print("the weights for max-sharpe-pf is {%1} and max annual return {%2}", opt_portfolio['x'], opt_portfolio['fun'])
#    show_optimal_portfolio(opt_portfolio, log_daily_returns, sample_portfolios)
#    show_CML(opt_portfolio, log_daily_returns, sample_portfolios)
    efficient_risk1 = efficient_risk_portfolio(sample_weights, log_daily_returns, 0.3)
    print(efficient_risk1['x'], sharpe_ratio(efficient_risk1['x'], log_daily_returns, 0.05) )
    show_all_portfolios(log_daily_returns, sample_portfolios, True,  show_CML=True, show_min_vol=True, show_efficient_risk=True, show_efficient_return=True)







