import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 網頁基本設定
st.set_page_config(page_title="AI 動能妖股雷達", page_icon="🚀", layout="wide")

# ==========================================
# 核心功能模組
# ==========================================
@st.cache_data(ttl=3600)
def get_tw_stock_list():
    stock_dict = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for m in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            df = pd.read_html(StringIO(res.text))[0].iloc[1:]
            for _, row in df.iterrows():
                try:
                    code_name = str(row[0]).split()
                    if len(code_name) == 2:
                        code, name = code_name
                        cat = str(row[4])
                        if len(code) == 4:
                            suffix = ".TW" if m == 2 else ".TWO"
                            stock_dict[f"{code}{suffix}"] = {"name": name, "sector": cat}
                except: continue
    except: pass
    return stock_dict

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'}
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(StringIO(res.text))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        names = df['Security'].tolist()
        sectors = df['GICS Sector'].tolist()
        return {t: {"name": n, "sector": s} for t, n, s in zip(tickers, names, sectors)}
    except: return {}

def check_market(symbol, ma_days=20):
    try:
        data = yf.download(symbol, period="50d", progress=False)
        close = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'].iloc[-1], pd.Series) else float(data['Close'].iloc[-1])
        ma20 = float(data['Close'].rolling(ma_days).mean().iloc[-1].iloc[0]) if isinstance(data['Close'].rolling(ma_days).mean().iloc[-1], pd.Series) else float(data['Close'].rolling(ma_days).mean().iloc[-1])
        return close >= ma20, close, ma20
    except:
        return True, 0, 0

# ==========================================
# 網頁介面設計
# ==========================================
st.title("🚀 AI 動能妖股雷達 (終極網頁版)")
st.markdown("自動掃描全市場，尋找符合【強勢動能】與【布林突破】的頂尖標的。")

# 側邊欄控制面板
st.sidebar.header("⚙️ 雷達設定")
market = st.sidebar.radio("選擇掃描市場", ["🇹🇼 台股 (上市/上櫃)", "🇺🇸 美股 (S&P 500)"])
st.sidebar.markdown("---")
st.sidebar.info("💡 **實戰紀律提醒**\n\n進場後若收盤跌破 5MA (五日線)，請無條件執行停損。")

if st.button("開始全面掃描", type="primary"):
    # 1. 大盤環境判斷
    status_placeholder = st.empty()
    status_placeholder.info("🌍 正在偵測大盤多空環境...")
    
    if "台股" in market:
        is_bull, idx_close, idx_ma = check_market("^TWII")
        stock_dict = get_tw_stock_list()
        vol_limit = 1000 # 台股千張
        vol_label = "5日均量(張)"
    else:
        is_bull, idx_close, idx_ma = check_market("^GSPC")
        stock_dict = get_sp500_tickers()
        vol_limit = 1000000 # 美股百萬股
        vol_label = "5日均量(M)"

    if is_bull:
        st.success(f"🟢 【大盤偏多】目前指數 ({idx_close:.2f}) 站上月線 ({idx_ma:.2f})，適合動能策略！")
    else:
        st.error(f"🔴 【大盤偏空】目前指數 ({idx_close:.2f}) 跌破月線 ({idx_ma:.2f})，極易假突破，建議空手觀望！")

    if not stock_dict:
        st.error("❌ 無法取得股票清單，請稍後再試。")
        st.stop()

    # 2. 開始掃描運算
    all_tickers = list(stock_dict.keys())
    progress_bar = st.progress(0)
    
    batch_size = 50
    records = []
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        status_placeholder.info(f"⛏️ 正在下載歷史數據與計算特徵... 進度: {min(i+batch_size, len(all_tickers))}/{len(all_tickers)}")
        progress_bar.progress(min(i+batch_size, len(all_tickers)) / len(all_tickers))
        
        try:
            data = yf.download(batch, period="100d", interval="1d", group_by='ticker', auto_adjust=False, progress=False, threads=True)
            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    if df.empty or len(df) < 65: continue
                    df = df.dropna()
                    
                    close = df['Close']
                    vol = df['Volume']
                    
                    # 成交量濾網
                    if "台股" in market:
                        avg_vol = float((vol.tail(5).mean()) / 1000)
                    else:
                        avg_vol = float(vol.tail(5).mean())
                        
                    if avg_vol < vol_limit: continue
                    
                    # 計算均線與特徵
                    ma5 = float(close.rolling(5).mean().iloc[-1])
                    ma20 = float(close.rolling(20).mean().iloc[-1])
                    ma60 = float(close.rolling(60).mean().iloc[-1])
                    current_close = float(close.iloc[-1])
                    
                    # 必須站在 5MA 之上
                    if current_close < ma5: continue
                    
                    daily_ret = close.pct_change()
                    hist_vol = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
                    std20 = close.rolling(20).std().iloc[-1]
                    bb_upper = float(ma20 + 2 * std20)
                    bb_width = float((bb_upper - (ma20 - 2 * float(std20))) / ma20 * 100)
                    
                    p_to_ma60 = (current_close / ma60 - 1) * 100
                    trend_str = (ma5 / ma60 - 1) * 100
                    p_to_ma20 = (current_close / ma20 - 1) * 100
                    p_to_bbupper = (current_close / bb_upper - 1) * 100
                    roc_10 = float((current_close - close.iloc[-11]) / close.iloc[-11] * 100)
                    
                    if np.isnan(hist_vol) or np.isnan(roc_10): continue

                    records.append({
                        'ID': ticker.replace(".TW", "").replace(".TWO", ""),
                        '股名': stock_dict[ticker]['name'],
                        '板塊產業': stock_dict[ticker]['sector'],
                        '收盤價': round(current_close, 2),
                        'MA5 (防守線)': round(ma5, 2),
                        vol_label: round(avg_vol / 1000000, 2) if "美股" in market else int(avg_vol),
                        'F_Hist_Vol': hist_vol, 'F_BB_Width': bb_width, 'F_P_to_MA60': p_to_ma60,
                        'F_Trend_Strength': trend_str, 'F_P_to_MA20': p_to_ma20, 'F_P_to_BBUpper': p_to_bbupper, 'F_ROC_10': roc_10
                    })
                except: continue
        except: pass

    # 3. AI 排名運算
    status_placeholder.success("✅ 數據下載完成！正在執行 AI 權重運算...")
    progress_bar.empty()
    
    df_res = pd.DataFrame(records)
    if not df_res.empty:
        features = ['F_Hist_Vol', 'F_BB_Width', 'F_P_to_MA60', 'F_Trend_Strength', 'F_P_to_MA20', 'F_P_to_BBUpper', 'F_ROC_10']
        weights = [29.08, 19.33, 10.39, 7.67, 7.26, 5.09, 4.25]

        for f in features: df_res[f + '_Rank'] = df_res[f].rank(pct=True)
        df_res['AI 總分'] = 0.0
        for f, w in zip(features, weights): df_res['AI 總分'] += df_res[f + '_Rank'] * w
        
        df_res['AI 總分'] = df_res['AI 總分'].round(2)
        top20 = df_res.sort_values(by='AI 總分', ascending=False).head(20)
        
        # 顯示結果表格 (隱藏中間計算過程)
        display_cols = ['ID', '股名', '板塊產業', '收盤價', 'MA5 (防守線)', vol_label, 'AI 總分']
        st.dataframe(top20[display_cols], use_container_width=True, hide_index=True)
        
        # 供下載的 CSV 包含所有特徵
        csv = top20.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載完整 CSV 報表",
            data=csv,
            file_name=f"Radar_Top20_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("目前沒有符合條件的標的。")