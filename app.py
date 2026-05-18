# ==========================================
# 🚀 終極實戰版 MANUS AI 動能妖股雷達 (Google Colab 專用)
# 優化重點：
# 1. 具備「流動性濾網」 (5日均量 > 1000張)
# 2. 具備「大盤多空環境濾網」 (自動判斷加權指數是否跌破月線)
# 3. 【新增】「MA5 乖離率」與「過熱動態扣分機制」 (乖離 > 5% 自動扣分)
# ==========================================

!pip install yfinance pandas numpy requests lxml -q

import sys, os, requests, urllib3
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import warnings
from google.colab import files

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("="*85)
print(" 🚀 終極實戰版動能雷達 啟動...")
print(" 🛡️ 內建安全機制：【大盤過濾】+【流動性過濾】+【乖離率扣分防護】")
print("="*85)

# ==========================================
# 🌍 步驟 0：判斷大盤多空環境
# ==========================================
def check_market_environment():
    print("🌍 [0/3] 正在偵測台股大盤多空環境...")
    try:
        twii = yf.download("^TWII", period="50d", progress=False)
        twii_close = float(twii['Close'].iloc[-1].iloc[0]) if isinstance(twii['Close'].iloc[-1], pd.Series) else float(twii['Close'].iloc[-1])
        twii_20ma = float(twii['Close'].rolling(20).mean().iloc[-1].iloc[0]) if isinstance(twii['Close'].rolling(20).mean().iloc[-1], pd.Series) else float(twii['Close'].rolling(20).mean().iloc[-1])
        
        is_bull = twii_close >= twii_20ma
        
        print("-" * 50)
        if is_bull:
            print(f"🟢 【大盤偏多】目前加權指數 ({twii_close:.0f}) 站上月線 ({twii_20ma:.0f})")
            print("👉 資金環境佳，動能策略勝率較高，可依名單適度佈局。")
        else:
            print(f"🔴 【大盤偏空】目前加權指數 ({twii_close:.0f}) 跌破月線 ({twii_20ma:.0f})")
            print("⚠️ 警告：空頭/震盪市場下，動能突破策略極易失效 (假突破多)！")
            print("👉 強烈建議：縮小資金部位，甚至【空手觀望】。")
        print("-" * 50 + "\n")
        return is_bull
    except Exception as e:
        print(f"⚠️ 大盤偵測失敗，請自行留意風險。({e})\n")
        return True 

# ==========================================
# 📋 抓取台股清單 (上市+上櫃)
# ==========================================
def get_tw_stock_list():
    print("📋 [1/3] 正在同步台股清單與產業資訊...")
    stock_dict = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for m in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            df = pd.read_html(res.text)[0].iloc[1:]
            for _, row in df.iterrows():
                try:
                    code_name = str(row[0]).split()
                    if len(code_name) == 2:
                        code, name = code_name
                        cat = str(row[4])
                        if len(code) == 4:
                            suffix = ".TW" if m == 2 else ".TWO"
                            stock_dict[f"{code}{suffix}"] = {"name": name, "industry": cat}
                except: continue
    except Exception as e: 
        print(f"❌ 抓取清單失敗: {e}")
    return stock_dict

# ==========================================
# 🚀 運算核心
# ==========================================
def main():
    is_bull_market = check_market_environment()

    stock_dict = get_tw_stock_list()
    all_tickers = list(stock_dict.keys())
    print(f"✅ 取得標的共 {len(stock_dict)} 檔。")
    print("⛏️ [2/3] 下載歷史數據並計算動能因子 (這需要幾分鐘的時間)...")
    
    batch_size = 50
    records = []
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"   進度: {min(i+batch_size, len(all_tickers))}/{len(all_tickers)}", end='\r')
        
        try:
            data = yf.download(batch, period="100d", interval="1d", group_by='ticker', auto_adjust=False, progress=False, threads=True)
            
            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    if df.empty or len(df) < 65: continue
                    df = df.dropna()
                    
                    close = df['Close']
                    vol = df['Volume']
                    
                    # 濾網：5日平均成交量 > 1000張
                    avg_vol_5d = float((vol.tail(5).mean()) / 1000)
                    if avg_vol_5d < 1000: continue
                    
                    ma5 = float(close.rolling(5).mean().iloc[-1])
                    ma20 = float(close.rolling(20).mean().iloc[-1])
                    ma60 = float(close.rolling(60).mean().iloc[-1])
                    
                    daily_ret = close.pct_change()
                    hist_vol = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
                    
                    std20 = close.rolling(20).std().iloc[-1]
                    bb_upper = float(ma20 + 2 * std20)
                    bb_width = float((bb_upper - (ma20 - 2 * float(std20))) / ma20 * 100)
                    
                    current_close = float(close.iloc[-1])
                    
                    p_to_ma60 = (current_close / ma60 - 1) * 100
                    trend_str = (ma5 / ma60 - 1) * 100
                    p_to_ma20 = (current_close / ma20 - 1) * 100
                    p_to_bbupper = (current_close / bb_upper - 1) * 100
                    roc_10 = float((current_close - close.iloc[-11]) / close.iloc[-11] * 100)
                    
                    # ✨ 【新增】計算 5MA 乖離率
                    ma5_bias = ((current_close - ma5) / ma5) * 100
                    
                    if np.isnan(hist_vol) or np.isnan(roc_10): continue

                    records.append({
                        'ID': ticker.replace(".TW", "").replace(".TWO", ""),
                        'Name': stock_dict[ticker]['name'],
                        'Industry': stock_dict[ticker]['industry'],
                        'Close': current_close,
                        'MA5': ma5,
                        'MA5_Bias': ma5_bias, # 寫入資料庫
                        'Avg_Vol_5D': avg_vol_5d,
                        'F_Hist_Vol': hist_vol,
                        'F_BB_Width': bb_width,
                        'F_P_to_MA60': p_to_ma60,
                        'F_Trend_Strength': trend_str,
                        'F_P_to_MA20': p_to_ma20,
                        'F_P_to_BBUpper': p_to_bbupper,
                        'F_ROC_10': roc_10
                    })
                except: continue
        except: pass

    print("\n✅ 特徵萃取完成！")
    print("🧠 [3/3] 執行 AI PR值排名權重與乖離率扣分運算...")
    
    df_res = pd.DataFrame(records)
    if df_res.empty:
        print("❌ 找不到符合條件的股票。")
        return

    features = ['F_Hist_Vol', 'F_BB_Width', 'F_P_to_MA60', 'F_Trend_Strength', 'F_P_to_MA20', 'F_P_to_BBUpper', 'F_ROC_10']
    weights = [29.08, 19.33, 10.39, 7.67, 7.26, 5.09, 4.25]

    for f in features:
        df_res[f + '_Rank'] = df_res[f].rank(pct=True)

    df_res['Base_AI_Score'] = 0.0
    for f, w in zip(features, weights):
        df_res['Base_AI_Score'] += df_res[f + '_Rank'] * w

    # ✨ 【新增】過熱動態扣分機制：乖離率大於 5% 的部分，每多 1% 扣 5 分
    df_res['Bias_Penalty'] = df_res['MA5_Bias'].apply(lambda x: (x - 5) * -5 if x > 5 else 0)
    df_res['Adj_AI_Score'] = df_res['Base_AI_Score'] + df_res['Bias_Penalty']

    # 濾網：只顯示收盤價站在 5MA 之上的股票
    df_filtered = df_res[df_res['Close'] >= df_res['MA5']].copy()
    
    # 改用「調整後 AI 總分」來進行最終排名
    top20 = df_filtered.sort_values(by='Adj_AI_Score', ascending=False).head(20)

    print("\n" + "="*85)
    print(f" 🎯 終極版 動能極限妖股 TOP 20 名單 ({datetime.now().strftime('%Y-%m-%d')})")
    print("="*85)
    
    if not is_bull_market:
        print(" 🚨 【再次提醒】目前為大盤偏空環境，操作請極度謹慎！")
        print("-" * 85)
        
    # 調整表格寬度以容納新資訊
    print(f"{'排名':<2} | {'代號':<4} | {'股名':<6} | {'收盤':<6} | {'MA5防守':<7} | {'乖離率%':<6} | {'AI總分':<6}")
    print("-" * 85)

    for i, (_, row) in enumerate(top20.iterrows(), 1):
        name = row['Name'][:4]
        # 特別標示危險的乖離率
        bias_str = f"{row['MA5_Bias']:>5.2f}%"
        if row['MA5_Bias'] > 5.0:
            bias_str = bias_str + "⚠️"
            
        print(f"{i:<4} | {row['ID']:<6} | {name:<8} | {row['Close']:>7.2f} | {row['MA5']:>9.2f} | {bias_str:<8} | {row['Adj_AI_Score']:>6.2f}")
        
    print("="*85)
    print("⚠️ 實戰提醒：\n 1. 🟢 乖離率 0~3%：首選完美試單點。\n 2. 🟡 乖離率 3~5%：注意追高風險。\n 3. 🔴 看到 ⚠️ 標記代表乖離率大於 5% 已扣分，請勿盲目追高。\n 4. 買進後若跌破 MA5 防守線，請務必無情停損。")
    print("="*85 + "\n")
    
    filename = f"AI_Top20_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # 整理輸出的 CSV 欄位順序
    export_cols = ['ID', 'Name', 'Industry', 'Close', 'MA5', 'MA5_Bias', 'Adj_AI_Score', 'Base_AI_Score', 'Bias_Penalty', 'Avg_Vol_5D']
    top20_export = top20[export_cols].rename(columns={
        'MA5_Bias': '5MA乖離率(%)', 
        'Adj_AI_Score': '調整後AI總分',
        'Base_AI_Score': '基礎AI分',
        'Bias_Penalty': '乖離懲罰分'
    })
    
    top20_export.to_csv(filename, index=False, encoding='utf-8-sig')
    files.download(filename)
    print(f"📁 報表已自動生成並下載：{filename}")

if __name__ == "__main__":
    main()
