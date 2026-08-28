import yfinance as yf

# Download historical data for a stock (e.g., Apple)
data = yf.download("AAPL", start="2018-01-01", end="2026-01-01")
# Drop missing values
data = data.dropna()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split data into Features (X) and Target (y)
X = data[['SMA_10', 'SMA_50', 'RSI', 'Volume']]
y = data['Target_Close_Price']

# Split chronologically (Do NOT shuffle stock data, as time order matters!)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

