import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 網頁介面設定
# ==========================================
st.set_page_config(page_title="AI 動能妖股雷達", layout="wide")
st.title("🚀 終極版 AI 動能妖股雷達 (台/美雙引擎)")
st.markdown("用量化數據捕捉資金動能，嚴守 5MA 防守紀律。")

# ==========================================
# 側邊欄設定 (市場切換與動態輸入)
# ==========================================
st.sidebar.header("⚙️ 參數設定")

# 新增：市場切換開關
market = st.sidebar.radio("🌍 選擇掃描市場：", ["台股 (TW)", "美股 (US)"])

# 根據選擇的市場，載入不同的預設股票與成交量名稱
if market == "台股 (TW)":
    default_tickers = "2330.TW, 2303.TW, 2317.TW, 3231.TW, 2382.TW, 3037.TW, 2351.TW, 8054.TWO, 6285.TW, 3017.TW, 3008.TW, 3443.TW, 3583.TW, 3661.TW, 2363.TW"
    vol_unit_name = "5日均量(張)"
else:
    # 美股熱門科技與動能股範例
    default_tickers = "NVDA, TSLA, AAPL, MSFT, AMD, META, AMZN, GOOGL, SMCI, ARM, PLTR, COIN, SOFI, PLUG, RIOT"
    vol_unit_name = "5日均量(百萬股)"

ticker_input = st.sidebar.text_area("請輸入要掃描的股票代號 (用逗號分隔)：", value=default_tickers, height=150)

# 解析輸入的股票代號
ticker_list = [t.strip() for t in ticker_input.split(',') if t.strip()]

# 設定資料抓取區間
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=60)

# ==========================================
# 核心運算函數
# ==========================================
@st.cache_data(ttl=3600) # 加上快取機制
def calculate_momentum_score(ticker, market_type):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty or len(df) < 20:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        current_close = float(df['Close'].iloc[-1])
        
        # 新增：依據市場別，處理成交量單位差異
        if market_type == "台股 (TW)":
            current_vol = float(df['Volume'].iloc[-1]) / 1000
            df['Vol_5MA'] = df['Volume'].rolling(window=5).mean() / 1000
        else:
            # 美股換算成「百萬股 (Millions)」方便閱讀
            current_vol = float(df['Volume'].iloc[-1]) / 1000000
            df['Vol_5MA'] = df['Volume'].rolling(window=5).mean() / 1000000
            
        vol_5ma = float(df['Vol_5MA'].iloc[-1])
        
        # 基礎濾網：台股<1000張剔除，美股<100萬股(1.0)剔除
        if (market_type == "台股 (TW)" and vol_5ma < 1000) or (market_type == "美股 (US)" and vol_5ma < 1.0):
            return None
            
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        ma5 = float(df['MA5'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
        df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
        
        df['Return'] = df['Close'].pct_change()
        volatility = float(df['Return'].rolling(window=10).std().iloc[-1]) * np.sqrt(252)
        
        ma5_bias = ((current_close - ma5) / ma5) * 100
        
        base_score = 0
        if current_close > ma5: base_score += 20
        if ma5 > ma20: base_score += 15
        if current_close > float(df['BB_Upper'].iloc[-2]): base_score += 25
        if volatility > 0.3: base_score += 20
        if current_vol > vol_5ma * 1.5: base_score += 20
        
        bias_penalty = 0
        if ma5_bias > 5:
            bias_penalty = (ma5_bias - 5) * -5 
            
        final_score = base_score + bias_penalty

        # 美股不需要去除 .TW 後綴，台股才需要
        ticker_name = ticker.replace('.TW', '').replace('.TWO', '') if market_type == "台股 (TW)" else ticker

        return {
            '股票代號': ticker_name,
            '收盤價': round(current_close, 2),
            'MA5(防守線)': round(ma5, 2),
            '5MA乖離率(%)': round(ma5_bias, 2),
            '調整後AI總分': round(final_score, 1),
            '基礎AI分': round(base_score, 1),
            '乖離懲罰分': round(bias_penalty, 1),
            # 美股顯示到小數點第一位(例如 5.2 百萬股)，台股顯示整數(例如 1500 張)
            vol_unit_name: round(vol_5ma, 1) if market_type == "美股 (US)" else int(vol_5ma)
        }
    except Exception as e:
        return None

# ==========================================
# 執行按鈕與畫面呈現
# ==========================================
if st.sidebar.button("啟動掃描 🎯"):
    with st.spinner(f"正在連線抓取【{market}】最新資料，請稍候..."):
        results = []
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            res = calculate_momentum_score(ticker, market)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(ticker_list))
            
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by='調整後AI總分', ascending=False).reset_index(drop=True)
            
            st.success(f"🎯 掃描完成！今日 {market} Top 20 狙擊榜單：")
            
            st.dataframe(
                df_results.head(20),
                use_container_width=True,
                height=600
            )
            
            st.info("""
            **實戰教練小提醒：**
            * 🟢 **乖離率 0% ~ 3%**：完美獵物，防守風險極低。
            * 🟡 **乖離率 3% ~ 5%**：注意追高風險，建議等回檔再試單。
            * 🔴 **乖離率 > 5%**：容易雙巴，不建議買進，除非隔天回測 5MA 有撐。
            """)
        else:
            st.warning("今日沒有符合條件的股票，或是資料抓取出現異常。")
