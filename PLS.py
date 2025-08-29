#####import 套件們 #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

#############計算PLS index###############
##匯入資料##
file_path = r"D:\NSYSU FIN\投資策略\for_github\merged_df_output.xlsx"
merged_df = pd.read_excel(file_path)

##定義變數##
target_var = 'RETURN_RETURN'
exclude_cols = ['YEAR', 'MONTH', 'Date', 'PLS_Index', 'Predicted_Return', target_var]
X_vars = [col for col in merged_df.columns if col not in exclude_cols]

##PLS model##
train_df = merged_df[X_vars + [target_var]].dropna()
X_full = train_df[X_vars]
y_full = train_df[[target_var]]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full)

pls = PLSRegression(n_components=1)
pls.fit(X_scaled, y_scaled)

# 擷取第一個 component 的 weights（β）
pls_weights = pls.x_weights_.flatten()
weight_dict = dict(zip(X_vars, pls_weights))

##計算每一期的PLS Index##
pls_index_list = []
for idx, row in merged_df.iterrows():
    sum_val = 0
    sum_wt2 = 0  # 為了 normalize

    for var in X_vars:
        val = row[var]
        wt = weight_dict[var]
        if pd.notna(val):
            sum_val += val * wt
            sum_wt2 += wt**2  # 可以選擇是否 normalize

    if sum_wt2 > 0:
        pls_index = sum_val / np.sqrt(sum_wt2)  # Normalize by used weights
    else:
        pls_index = np.nan

    pls_index_list.append(pls_index)

##匯出##
merged_df['PLS_Index'] = pls_index_list
merged_df.to_excel(r"D:\NSYSU FIN\投資策略\for_github\merged_with_pls_ignore_na.xlsx", index=False)

pls_weights = pls.x_weights_.flatten()# 提取第一主成分的權重
weight_df = pd.DataFrame({
    'Variable': X_vars,
    'PLS_Weight': pls_weights
})

weight_df['Abs_Weight'] = weight_df['PLS_Weight'].abs()
weight_df = weight_df.sort_values(by='Abs_Weight', ascending=False)# 按絕對值排序，越大越好

#print(weight_df[['Variable', 'PLS_Weight']].head(10))

#########繪製pls趨勢圖#########
file_path = r"D:\NSYSU FIN\投資策略\for_github\merged_with_pls_ignore_na.xlsx"
df = pd.read_excel(file_path)

scaler = StandardScaler()
df['PLS_Index'] = scaler.fit_transform(df[['PLS_Index']])

df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2))

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['PLS_Index'], label='PLS-Based Uncertainty Index', color='blue')

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Monthly PLS-Based Uncertainty Index Trend')
plt.xlabel('Date')
plt.ylabel('PLS Index')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#########跑pls index是否具有預測效果#########
df['PLS_Index_LAG'] = df['PLS_Index'].shift(1) #lag 1 period
reg_df = df.dropna(subset=['PLS_Index_LAG', 'RETURN_RETURN'])

##create X and y##
X = sm.add_constant(reg_df['PLS_Index_LAG'])  # 加入截距項
y = reg_df['RETURN_RETURN']

##回歸##
model = sm.OLS(y, X).fit()
print(model.summary())

#########滾動視窗 out-of-sample forecast#########
rolling_predictions = []
rolling_actual = []
rolling_dates = [] # 準備儲存預測報酬結果

min_obs = 36 # 至少 3 年訓練資料，因為我是月資料

for i in range(min_obs, len(df)-1):  #滾動 windows，並預測下一期
    train_df = df.iloc[:i].dropna(subset=['PLS_Index_LAG', 'RETURN_RETURN'])
    test_df = df.iloc[i+1]  

    if pd.isna(test_df['PLS_Index_LAG']) or pd.isna(test_df['RETURN_RETURN']):
        continue

    X_train = sm.add_constant(train_df['PLS_Index_LAG'])
    y_train = train_df['RETURN_RETURN']

    model = sm.OLS(y_train, X_train).fit()

    X_test = [1, test_df['PLS_Index_LAG']]
    y_pred = model.predict(X_test)[0]

    rolling_predictions.append(y_pred)
    rolling_actual.append(test_df['RETURN_RETURN'])
    rolling_dates.append(f"{int(test_df['YEAR'])}-{int(test_df['MONTH']):02d}")

rolling_df = pd.DataFrame({
    'Date': rolling_dates,
    'Predicted_Return': rolling_predictions,
    'Actual_Return': rolling_actual
}) #用pls預測的股價報酬與實際做比較

##畫圖##
plt.figure(figsize=(12, 5))
plt.plot(rolling_df['Date'], rolling_df['Predicted_Return'], label='Predicted', color='blue')
plt.plot(rolling_df['Date'], rolling_df['Actual_Return'], label='Actual', color='orange')
step = 20 # 每隔 20 個月顯示一個 x 軸標籤
plt.xticks(rolling_df['Date'][::step], rotation=45)
plt.title('Rolling Out-of-Sample Forecast: Return')
plt.legend()
plt.tight_layout()
plt.show()

##低估與高估情況##
rolling_df['Diff'] = rolling_df['Actual_Return'] - rolling_df['Predicted_Return']

num_positive = (rolling_df['Diff'] > 0).sum()  # 差異 > 0，表示低估
num_negative = (rolling_df['Diff'] < 0).sum()  # 差異 < 0，表示高估
num_zero = (rolling_df['Diff'] == 0).sum()     # 剛好預測到位

print(f"正 (低估) 的數量: {num_positive}")
print(f"負 (高估) 的數量: {num_negative}")
print(f"無差異 的數量: {num_zero}")