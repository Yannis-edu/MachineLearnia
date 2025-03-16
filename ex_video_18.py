import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bitcoin = pd.read_csv("data/btc-usd-max.csv", index_col="snapped_at", parse_dates=True)
ethereum = pd.read_csv("data/eth-usd-max.csv", index_col="snapped_at", parse_dates=True)

bitcoin = bitcoin.drop(["market_cap", "total_volume"], axis=1)
ethereum = ethereum.drop(["market_cap", "total_volume"], axis=1)

bitcoin["buy"] = np.zeros(len(bitcoin))
bitcoin["sell"] = np.zeros(len(bitcoin))

bitcoin["rolling_max"] = bitcoin["price"].shift(1).rolling(window=28).max()
bitcoin["rolling_min"] = bitcoin["price"].shift(1).rolling(window=28).min()

bitcoin.loc[bitcoin["price"] > bitcoin["rolling_max"], "buy"] = 1
bitcoin.loc[bitcoin["price"] < bitcoin["rolling_min"], "sell"] = -1

plt.subplot(2, 1, 1)
plt.plot(bitcoin.loc["2019", "price"], c="orange")
plt.plot(ethereum.loc["2019", "price"], c="lightblue")

plt.plot(bitcoin.loc["2019", "rolling_max"], c="gray")
plt.plot(bitcoin.loc["2019", "rolling_min"], c="gray")

plt.subplot(2, 1, 2)
plt.plot(bitcoin.loc["2019", "buy"], c="green")
plt.plot(bitcoin.loc["2019", "sell"], c="red")

plt.show()

# bitcoin.diff().plot()
# plt.show()
