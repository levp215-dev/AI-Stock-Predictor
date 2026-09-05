# python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import date

# Download historical data
data = yf.download("AAPL", start="2018-01-01", end=date.today().strftime("%Y-%m-%d"))

# Flatten MultiIndex columns (yfinance can return MultiIndex for some setups)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join([str(x) for x in col if x is not None]).strip() for col in data.columns.values]

# Helper to find a column containing a keyword (case-insensitive)
def find_col(df, keyword):
    keyword = keyword.lower()
    for c in df.columns:
        if keyword in str(c).lower():
            return c
    return None

close_col = find_col(data, 'close')
volume_col = find_col(data, 'volume')

if close_col is None or volume_col is None:
    raise RuntimeError(f"Could not find required columns in data.columns: {list(data.columns)}")

# Ensure numeric
data[close_col] = pd.to_numeric(data[close_col], errors='coerce')
data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce')

# Compute indicators
close = data[close_col]
data['SMA_10'] = close.rolling(window=10).mean()
data['SMA_50'] = close.rolling(window=50).mean()

# RSI (14 period)
delta = close.diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.rolling(window=14).mean()
roll_down = down.rolling(window=14).mean()
rs = roll_up / roll_down
data['RSI'] = 100 - (100 / (1 + rs))

# Target: next-day close
data['Target'] = close.shift(-1)

# Drop rows with NaNs introduced by indicators or target
data = data.dropna()

# Features and target
X = data[['SMA_10', 'SMA_50', 'RSI', volume_col]]
y = data['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print("R^2 on test set:", score)

# Predict next day's close
last_row = data.iloc[-1][['SMA_10', 'SMA_50', 'RSI', volume_col]].values.reshape(1, -1)
next_day_prediction = model.predict(last_row)
print("Predicted next day's close:", next_day_prediction[0])

