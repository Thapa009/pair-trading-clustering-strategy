# pair-trading-clustering-strategy
"Pair Trading Clustering Strategy: Python code for clustering-based pair trading strategy on stock returns using K-Means, DBSCAN, Agglomerative clustering, retrieving historical data via Yahoo Finance API."

# Pair Trading Clustering Strategy

This Python program implements a pair trading strategy based on clustering analysis of stock returns. It utilizes K-Means, DBSCAN, and Agglomerative clustering algorithms to identify clusters of stocks, and then applies a pair trading strategy within each cluster. The historical stock data is retrieved using Yahoo Finance API (`yfinance`).

## Usage

1. Install necessary dependencies by running `pip install -r requirements.txt`.
2. Run the script `pair_trading.py`.
3. View the results including clustering summaries, trades executed, and final portfolio value.

## Dependencies

- yfinance
- pandas
- numpy
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
