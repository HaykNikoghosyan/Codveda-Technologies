import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
import numpy as np
np.random.seed(42)
values = 100 + np.linspace(0, 50, 365) + np.sin(np.linspace(0, 20, 365)) * 10 + np.random.normal(0, 2, 365)

df = pd.DataFrame({'Value': values}, index=dates)

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Original Data', color='blue')
plt.title('Time Series Data Plot', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('time_series_raw.png', dpi=300, bbox_inches='tight')
plt.show()

df['Moving_Avg'] = df['Value'].rolling(window=30).mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Original', alpha=0.5)
plt.plot(df.index, df['Moving_Avg'], label='30-Day Moving Average', color='red', linewidth=2)
plt.title('Time Series with Moving Average Smoothing', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('time_series_moving_avg.png', dpi=300, bbox_inches='tight')
plt.show()

decomposition = seasonal_decompose(df['Value'], model='additive', period=30)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle('Time Series Decomposition', fontsize=16)
plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()