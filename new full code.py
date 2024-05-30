import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Step 1: Retrieve List of NASDAQ Tickers
def get_nasdaq_tickers():
    
    nasdaq_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "NVDA", "NFLX", "INTC", "ADBE"]
    return nasdaq_tickers

tickers = get_nasdaq_tickers()
print(f"Tickers: {tickers}")

# Step 2: Retrieve Historical Data
def get_historical_data(tickers, start_date, end_date):
    print(f"Fetching historical data from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

start_date = '2010-01-01'
end_date = '2018-12-31'
historical_data = get_historical_data(tickers, start_date, end_date)
print(f"Historical data retrieved: {historical_data.shape}")

# Step 3: Clustering
returns = historical_data.pct_change().dropna()
print(f"Returns calculated: {returns.shape}")

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(returns.T)
returns['kmeans_cluster'] = kmeans.labels_

# DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=2).fit(returns.T)
returns['dbscan_cluster'] = dbscan.labels_

# Agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3).fit(returns.T)
returns['agglo_cluster'] = agglo.labels_

# Clustering summary
def clustering_summary(labels):
    unique, counts = np.unique(labels, return_counts=True)
    clusters = dict(zip(unique, counts))
    return clusters

print("K-Means Clustering:", clustering_summary(kmeans.labels_))
print("DBSCAN Clustering:", clustering_summary(dbscan.labels_))
print("Agglomerative Clustering:", clustering_summary(agglo.labels_))

# Step 4: Trading Strategy
class PairTradingStrategy:
    def __init__(self, historical_data, cluster_labels):
        self.data = historical_data
        self.cluster_labels = cluster_labels
        self.portfolio = {}
        self.cash = 100000  # Initial cash
        self.trades = []

    def trade_pair(self, pair, date):
        stock1, stock2 = pair
        if stock1 not in self.data.columns or stock2 not in self.data.columns:
            return

        price1 = self.data.at[date, stock1]
        price2 = self.data.at[date, stock2]

        spread_mean = (self.data[stock1] - self.data[stock2]).mean()
        spread_std = (self.data[stock1] - self.data[stock2]).std()

        spread = price1 - price2
        if spread > spread_mean + 2 * spread_std:
            self.portfolio[stock1] = self.portfolio.get(stock1, 0) - 100
            self.portfolio[stock2] = self.portfolio.get(stock2, 0) + 100
            self.cash += (price1 - price2) * 100
            self.trades.append((date, pair, 'short-long', price1, price2))
        elif spread < spread_mean - 2 * spread_std:
            self.portfolio[stock1] = self.portfolio.get(stock1, 0) + 100
            self.portfolio[stock2] = self.portfolio.get(stock2, 0) - 100
            self.cash -= (price1 - price2) * 100
            self.trades.append((date, pair, 'long-short', price1, price2))

    def handle_suspensions_and_delistings(self, date):
        for stock in list(self.portfolio.keys()):
            if stock not in self.data.columns:
                continue
            price = self.data.at[date, stock]
            if pd.isna(price):
                self.portfolio[stock] = 0

    def rebalance(self, date):
        for stock, quantity in list(self.portfolio.items()):
            price = self.data.at[date, stock]
            if not pd.isna(price):
                self.cash += price * quantity
        self.portfolio = {}

    def run_backtest(self, start_date, end_date, rebalance_period='monthly'):
        date_range = pd.date_range(start_date, end_date, freq='B')  # 'B' frequency is for business days
        
        for date in date_range:
            self.handle_suspensions_and_delistings(date)
            cluster_pairs = [(x, y) for x in self.cluster_labels.index for y in self.cluster_labels.index if x != y]
            for pair in cluster_pairs:
                self.trade_pair(pair, date)
            if (rebalance_period == 'monthly' and date.day == 1) or \
               (rebalance_period == 'weekly' and date.weekday() == 0):
                self.rebalance(date)

        final_data = self.data.loc[end_date]
        portfolio_value = self.cash
        for stock, quantity in self.portfolio.items():
            price = final_data.get(stock, 0)
            if not pd.isna(price):
                portfolio_value += price * quantity

        return portfolio_value, self.trades

# Running the backtest
cluster_labels = returns['kmeans_cluster']
strategy = PairTradingStrategy(historical_data, cluster_labels)
final_value, trades = strategy.run_backtest(start_date, end_date, rebalance_period='monthly')

print(f"Final portfolio value: ${final_value}")
print("Trades executed:")
for trade in trades:
    print(trade)
