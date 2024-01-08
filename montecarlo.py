# brownian motion: 
#   expect share price (S) to increase, ex. double rate, will be exponential growth rate
#   S = Se^(at)  -> a is alpha, b is beta, e is eulers number
#   brownian motion is oscillating up and down but same shape as exponential graph
#   S = S * e^((at) + beta * B))  -> exponential growth + some constant (beta is key variable)
#   (share price at t) = ...
#   solving will give natural log on one side = at + bB
#   note that B of t = standard normal distribution with mean 0 and variance t = N(0,t)
#   at + bB ~ N(at,(b^2)t)   ...    Var(X) = variance of a random variable X = a, then Var(kX) = (k^2)a
#   "alpha t + beta t is normally distributed by a mean of 0 + alpha t, and variance beta squared t"
#   natural log of S t over S 0 = N(at, (b^2)t)
#   share price at time t, share time at beginning is Log - Normal
#   log normal curve is a normal curve except skewed

# stock data is from Yahoo Finance csv
# Open: This is a stock’s initial price at the start of the trading day.
# Previous close:  This is the stock the priced closed at for the preceding trading day.
# High: The high represents a stock’s highest trading price for the day.
# Low: The low is a stock’s lowest trading price for the day.

import math # math
import pandas as pd # csv and data management
import numpy as np # std normal random number for brownian motion
import matplotlib.pyplot as plt # plotting

# load csv with pandas
df = pd.read_csv('DIS.csv') # df is dataframe
# print(df.to_string()) 

# compute daily returns
# daily return = close price - open price
# save the sum to calculate average/standard deviation
sum = 0
for i in range(len(df)):
    df.loc[i, 'daily_return'] = df.loc[i, 'Close'] - df.loc[i, 'Open']
    sum += df.loc[i, 'daily_return']

# estimate the drift and volatility of the stock from the historical data:
#   drift can be estimated as the mean daily return
drift = sum / len(df)
print("Drift: ", drift)

sum_squared_diff = 0
for i in range(len(df)):
    squared_diff = (df.loc[i, 'daily_return'] - drift) ** 2
    sum_squared_diff += squared_diff

# Calculate the volatility
volatility = np.sqrt(sum_squared_diff / len(df))
print("Volatility: ", volatility)

# Monte Carlo Simulation: For each simulation:
#   Start with the last known stock price.
#   Use the geometric Brownian motion formula to estimate the stock price for each day in the next year.
#   Record the path of the stock price.

# last known stock price
stock_initial = df.loc[len(df) - 1, 'Close'] # assumes csv files are loaded with most recent data at the bottom
# drift and volatility already calculated

def simulate_geometric_brownian_motion(S0, mu, sigma, T, n):
    # note: using different formula than the one given in the assigment yields better results
    
    # S0: init stock price
    # mu: drift (average return)
    # sigma: volatility (standard devitaion of return)
    # T: time period in years
    # n: number of time steps

    dt = T / n
    # generate random values from standard normal distribution
    rand_values = np.random.normal(0, 1, n)

    t = np.linspace(0, T, n+1) # linspace = linearly spaced array
    S = [S0]
 
    # simulate stock price path
    for i in range(1, n+1):
        S_next = S[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_values[i-1])
        S.append(S_next)

    # t: list of time points.
    # S: list of simulated stock prices.
    return t, S

S0 = stock_initial
mu = drift
sigma = volatility
T = 1.0  # time in years
n = 365  # number of trading days in the year

# create the plot
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Simulated Geometric Brownian Motion')

# calculate probability stock price exceeds a given value
print("Input threshold: ")
threshold = float(input())
count = 0
for i in range(10000):
    t, S = simulate_geometric_brownian_motion(S0, mu, sigma, T, n)
    # look at the last value in the list
    if S[n] > threshold:
        count += 1
    plt.plot(t, S)
print("Probability: ", count / 10000)
plt.show()




