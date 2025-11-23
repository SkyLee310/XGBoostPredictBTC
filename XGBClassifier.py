print(">>> Running XGBoost Classification Version ")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def clean_bitcoin_data(df):
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype(float)
    df["Change %"] = df["Change %"].str.replace("%", "", regex=False).astype(float)

    def convert_vol(value):
        value = value.replace(",", "")
        if "K" in value:
            return float(value.replace("K", "")) * 1000
        elif "M" in value:
            return float(value.replace("M", "")) * 1_000_000
        elif "B" in value:
            return float(value.replace("B", "")) * 1_000_000_000
        else:
            return float(value)

    df["Vol."] = df["Vol."].apply(convert_vol)
    return df

btc_file_path = 'Bitcoin Historical Data 5y.csv'
btc_data = pd.read_csv(btc_file_path)
btc_data = clean_bitcoin_data(btc_data)

btc_data["Price_lag1"] = btc_data["Price"].shift(1)
btc_data["Price_lag3"] = btc_data["Price"].shift(3)
btc_data["Price_lag7"] = btc_data["Price"].shift(7)
btc_data["Price_lag14"] = btc_data["Price"].shift(14)
btc_data["MA7"] = btc_data["Price"].rolling(7).mean()
btc_data["MA30"] = btc_data["Price"].rolling(30).mean()
btc_data["MA_Ratio"] = btc_data["MA7"] / btc_data["MA30"]
btc_data["Return_1d"] = btc_data["Price"].pct_change()
btc_data["Volatility_7d"] = btc_data["Price"].pct_change().rolling(7).std()


btc_data["Target"] = (btc_data["Price"].shift(-1) > btc_data["Price"]).astype(int)


btc_data = btc_data.dropna()

features = [
    "Open", "High", "Low", "Vol.", "Change %",
    "Price_lag1", "Price_lag3", "Price_lag7", "Price_lag14",
    "MA7", "MA30", "MA_Ratio",
    "Volatility_7d", "Return_1d"
]
X = btc_data[features]
y = btc_data["Target"]


split_index = int(len(X) * 0.8)
train_X, val_X = X.iloc[:split_index], X.iloc[split_index:]
train_y, val_y = y.iloc[:split_index], y.iloc[split_index:]


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=2,
    random_state=1,
    eval_metric="logloss"
)

model.fit(train_X, train_y)


val_preds = model.predict(val_X)

accuracy = accuracy_score(val_y, val_preds)
print("Validation Accuracy:", accuracy)
print(classification_report(val_y, val_preds))

joblib.dump(model, "btc_strategy.pkl")

print("模型已成功保存为 btc_strategy.pkl")
'''
plt.figure(figsize=(10, 4))
plt.plot(val_y.values[:50], label="True (0=Down,1=Up)", marker='o')
plt.plot(val_preds[:50], label="Predicted", marker='x')
plt.title("True vs Predicted (First 50 Days in Validation Set)")
plt.legend()
plt.show()
'''

#--------------------------------------Backtest-----------------------------------#

# 取验证集对应的真实价格
val_prices = btc_data["Price"].iloc[len(train_X):].reset_index(drop=True)

# 计算每日真实收益率
price_returns = val_prices.pct_change().fillna(0)

# 参数设置
threshold = 0.045    
position_size = 0.5 # 固定 50% 仓位
TP = 0.05           # 止盈 5%
SL = 0.02           # 止损 2%

strategy_returns = []

for i in range(len(price_returns)):
    daily_ret = price_returns[i]
    pred = val_preds[i]  # 1 = 做多, 0 = 做空

    # 不满足阈值 -> 不交易
    if abs(daily_ret) < threshold:
        strategy_returns.append(0)
        continue

    # 做多情况
    if pred == 1:
        if daily_ret >= TP:
            strategy_returns.append(TP * position_size)  # 止盈
        elif daily_ret <= -SL:
            strategy_returns.append(-SL * position_size)  # 止损
        else:
            strategy_returns.append(daily_ret * position_size)

    # 做空情况
    else:
        if daily_ret <= -TP:
            strategy_returns.append(TP * position_size)  # 空头止盈
        elif daily_ret >= SL:
            strategy_returns.append(-SL * position_size)  # 空头止损
        else:
            strategy_returns.append(-daily_ret * position_size)

# 转为 Series
strategy_returns = pd.Series(strategy_returns)
hold_returns = price_returns  # 对照组：全程持有

# 计算累计收益
strategy_cum = (1 + strategy_returns).cumprod()
hold_cum = (1 + hold_returns).cumprod()

# 输出结果
print("\n===== Enhanced Backtest (TP/SL + Position + Threshold) =====")
print(f"Strategy Final Return: {(strategy_cum.iloc[-1] - 1) * 100:.2f}%")
print(f"Buy & Hold Return: {(hold_cum.iloc[-1] - 1) * 100:.2f}%")

# Sharpe Ratio
risk_free_rate = 0
sharpe_ratio = (strategy_returns.mean() - risk_free_rate) / (strategy_returns.std() + 1e-9)
sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)

print(f"Daily Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Annualized Sharpe Ratio: {sharpe_ratio_annualized:.4f}")

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(strategy_cum, label="Strategy (Enhanced)", linewidth=2)
plt.plot(hold_cum, label="Buy & Hold", linestyle="--")
plt.title("Enhanced Strategy vs Buy & Hold")
plt.xlabel("Days")
plt.ylabel("Cumulative Return (Multiplier)")
plt.legend()
plt.grid(True)
plt.show()

#----------------------------------Equity Curve---------------------------------------------
# === 累计收益 ===
strategy_cum = (1 + strategy_returns).cumprod()
hold_cum = (1 + hold_returns).cumprod()

# === 最大回撤计算 ===
strategy_roll_max = strategy_cum.cummax()
drawdown = (strategy_cum - strategy_roll_max) / strategy_roll_max

# === 绘图 ===
plt.figure(figsize=(12,6))

# 资金曲线
plt.subplot(2,1,1)
plt.plot(strategy_cum, label="Strategy (ETH)", linewidth=2)
plt.plot(hold_cum, label="Buy & Hold (ETH)", linestyle="--")
plt.title("Cumulative Return Curve (ETH)")
plt.ylabel("Return (Multiplier)")
plt.legend()
plt.grid(True)

# 回撤曲线
plt.subplot(2,1,2)
plt.plot(drawdown, color="red")
plt.title("Drawdown (ETH Strategy)")
plt.ylabel("Drawdown")
plt.xlabel("Days")
plt.grid(True)

plt.tight_layout()
plt.show()

# === 输出回撤指标 ===
max_drawdown = drawdown.min()
print(f"Max Drawdown (Strategy): {max_drawdown:.2%}")