from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'Data'

SYMBOL = 'AAPL'

stocks = pd.read_csv(DATA_DIR / '2) Stock Prices Data Set.csv', parse_dates=['date'])
df = (stocks[stocks['symbol'] == SYMBOL]
      .sort_values('date')
      .set_index('date')[['close']]
      .rename(columns={'close': 'Value'}))

print(f'{SYMBOL}: {len(df)} trading days, {df.index.min().date()} - {df.index.max().date()}')

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Close Price', color='blue')
plt.title(f'{SYMBOL} Closing Price', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.savefig(BASE_DIR / 'time_series_raw.png', dpi=300, bbox_inches='tight')
plt.show()

df['Moving_Avg'] = df['Value'].rolling(window=30).mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Original', alpha=0.5)
plt.plot(df.index, df['Moving_Avg'], label='30-Day Moving Average', color='red', linewidth=2)
plt.title(f'{SYMBOL} Close Price with Moving Average Smoothing', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / 'time_series_moving_avg.png', dpi=300, bbox_inches='tight')
plt.show()

# ~252 trading days in a year -> yearly seasonality
decomposition = seasonal_decompose(df['Value'], model='additive', period=252)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle(f'{SYMBOL} Time Series Decomposition', fontsize=16)
plt.savefig(BASE_DIR / 'time_series_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()
