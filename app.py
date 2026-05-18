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
                except Exception as e: 
                    print(f"解析台股 {row[0]} 發生錯誤: {e}")
                    continue
    except Exception as e: 
        print(f"獲取台股清單發生錯誤: {e}")
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
    except Exception as e: 
        print(f"獲取美股清單發生錯誤: {e}")
        return {}

def check_market(symbol, ma_days=20):
    try:
        data = yf.download(symbol, period="50d", progress=False)
        close = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'].iloc[-1], pd.Series) else float(data['Close'].iloc[-1])
        ma20 = float(data['Close'].rolling(ma_days).mean().iloc[-1].iloc[0]) if isinstance(data['Close'].rolling(ma_days).mean().iloc[-1], pd.Series) else float(data['Close'].rolling(ma_days).mean().iloc[-1])
        return close >= ma20, close, ma20
    except Exception as e:
        print(f"大盤檢查錯誤 ({symbol}): {e}")
        return True, 0, 0

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_calculate_features(market_name):
    if "台股" in market_name:
        stock_dict = get_tw_stock_list()
        vol_label = "5日均量(張)"
    else:
        stock_dict = get_sp500_tickers()
        vol_label = "5日均量(M)"

    if not stock_dict:
        return pd.DataFrame(), vol_label

    all_tickers = list(stock_dict.keys())
    batch_size = 50
    records = []
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="100d", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    if df.empty or len(df) < 65: continue
                    df = df.dropna()
                    
                    close = df['Close']
                    vol = df['Volume']
                    high = df['High']
                    low = df['Low']
                    open_p = df['Open']
                    
                    # 取出最新一日的 K 線數據
                    c_close = float(close.iloc[-1])
                    c_open = float(open_p.iloc[-1])
                    c_high = float(high.iloc[-1])
                    c_low = float(low.iloc[-1])
                    
                    # 🛡️ 升級 1：避雷針過濾 (上影線比例 > 50% 絕對不碰)
                    k_len = c_high - c_low
                    if k_len > 0:
                        upper_shadow = (c_high - max(c_open, c_close)) / k_len
                        if upper_shadow > 0.5:
                            continue # 主力拉高出貨，直接從榜單無情剔除
                            
                    # ⚡ 升級 2：效能優化，避免重複 rolling
                    roll_5 = close.rolling(5)
                    roll_20 = close.rolling(20)
                    roll_60 = close.rolling(60)
                    
                    ma5 = float(roll_5.mean().iloc[-1])
                    if c_close < ma5: continue # 必須站在 5MA 之上
                    
                    ma5_bias = ((c_close - ma5) / (ma5 + 1e-9)) * 100
                    ma20 = float(roll_20.mean().iloc[-1])
                    ma60 = float(roll_60.mean().iloc[-1])
                    
                    if "台股" in market_name:
                        avg_vol = float((vol.tail(5).mean()) / 1000)
                    else:
                        avg_vol = float(vol.tail(5).mean())
                    
                    # 📈 升級 3：新增爆量倍數因子 (當日量 / 20日均量)
                    vol_20_mean = float(vol.tail(20).mean())
                    vol_ratio = float(vol.iloc[-1]) / (vol_20_mean + 1e-9)
                    
                    daily_ret = close.pct_change()
                    hist_vol = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
                    std20 = float(roll_20.std().iloc[-1])
                    bb_upper = float(ma20 + 2 * std20)
                    bb_width = float((bb_upper - (ma20 - 2 * std20)) / (ma20 + 1e-9) * 100)
                    
                    p_to_ma60 = (c_close / (ma60 + 1e-9) - 1) * 100
                    trend_str = (ma5 / (ma60 + 1e-9) - 1) * 100
                    p_to_ma20 = (c_close / (ma20 + 1e-9) - 1) * 100
                    p_to_bbupper = (c_close / (bb_upper + 1e-9) - 1) * 100
                    roc_10 = float((c_close - close.iloc[-11]) / (close.iloc[-11] + 1e-9) * 100)
                    
                    if np.isnan(hist_vol) or np.isnan(roc_10): continue

                    records.append({
                        'ID': ticker.replace(".TW", "").replace(".TWO", ""),
                        '股名': stock_dict[ticker]['name'],
                        '板塊產業': stock_dict[ticker]['sector'],
                        '收盤價': round(c_close, 2),
                        'MA5 (防守線)': round(ma5, 2),
                        '5MA乖離率(%)': round(ma5_bias, 2),
                        '爆量倍數': round(vol_ratio, 2), # 新增展示欄位
                        'Avg_Vol': avg_vol, 
                        vol_label: round(avg_vol / 1000000, 2) if "美股" in market_name else int(avg_vol),
                        'F_Vol_Ratio': vol_ratio, # 新增 AI 評分特徵
                        'F_Hist_Vol': hist_vol, 'F_BB_Width': bb_width, 'F_P_to_MA60': p_to_ma60,
                        'F_Trend_Strength': trend_str, 'F_P_to_MA20': p_to_ma20, 'F_P_to_BBUpper': p_to_bbupper, 'F_ROC_10': roc_10
                    })
                except Exception as e: 
                    continue
        except Exception as e:
            pass
    
    return pd.DataFrame(records), vol_label

# ==========================================
# 網頁介面設計
# ==========================================
st.title("🚀 AI 動能妖股雷達 (終極實戰版)")
st.markdown("內建【避雷針過濾】與【爆量引擎】，自動尋找最純粹的動能妖股。")

st.sidebar.header("⚙️ 雷達設定")
market = st.sidebar.radio("選擇掃描市場", ["🇹🇼 台股 (上市/上櫃)", "🇺🇸 美股 (S&P 500)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 策略微調")
user_vol_limit = st.sidebar.number_input("最小均量限制 (台:張 / 美:百萬股)", min_value=100, max_value=20000, value=1000, step=100)
user_bias_limit = st.sidebar.slider("乖離率扣分門檻 (%)", min_value=1, max_value=15, value=5)
user_penalty = st.sidebar.number_input("超過門檻每 1% 扣幾分?", min_value=1, max_value=20, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.info("💡 **實戰紀律提醒**\n\n進場後若收盤跌破 5MA (五日線)，請無條件執行停損。")

if st.button("開始全面掃描", type="primary"):
    if "台股" in market:
        is_bull, idx_close, idx_ma = check_market("^TWII")
    else:
        is_bull, idx_close, idx_ma = check_market("^GSPC")

    if is_bull:
        st.success(f"🟢 【大盤偏多】目前指數 ({idx_close:.2f}) 站上月線 ({idx_ma:.2f})，適合動能策略！")
    else:
        st.error(f"🔴 【大盤偏空】目前指數 ({idx_close:.2f}) 跌破月線 ({idx_ma:.2f})，極易假突破，建議空手觀望！")

    with st.status(f"🔍 啟動 {market} 運算中 (若無快取約需 1-2 分鐘)...", expanded=True) as status:
        df_all, vol_label = fetch_and_calculate_features(market)
        
        if df_all.empty:
            status.update(label="❌ 掃描失敗或無符合標的", state="error", expanded=False)
            st.error("目前無法取得數據，請稍後再試。")
            st.stop()
            
        df_records = df_all[df_all['Avg_Vol'] >= user_vol_limit].copy()
        
        if df_records.empty:
            status.update(label="❌ 無符合條件的標的", state="error", expanded=False)
            st.warning(f"目前沒有任何標的的成交量大於 {user_vol_limit}，請嘗試調低標準。")
            st.stop()

        # 🎯 升級 4：加入爆量倍數參與 AI 評分 (重新分配權重，總計 100)
        features = ['F_Vol_Ratio', 'F_Hist_Vol', 'F_BB_Width', 'F_P_to_MA60', 'F_Trend_Strength', 'F_P_to_MA20', 'F_P_to_BBUpper', 'F_ROC_10']
        weights = [15.00, 24.00, 16.00, 10.00, 10.00, 10.00, 10.00, 5.00] 

        for f in features: df_records[f + '_Rank'] = df_records[f].rank(pct=True)
        
        df_records['AI 總分'] = 0.0
        for f, w in zip(features, weights): df_records['AI 總分'] += df_records[f + '_Rank'] * w
        
        df_records['乖離懲罰分'] = df_records['5MA乖離率(%)'].apply(lambda x: (x - user_bias_limit) * -user_penalty if x > user_bias_limit else 0)
        df_records['AI 總分'] = df_records['AI 總分'] + df_records['乖離懲罰分']
        df_records['AI 總分'] = df_records['AI 總分'].round(2)
        
        top20 = df_records.sort_values(by='AI 總分', ascending=False).head(20)
        
        status.update(label="✅ 掃描與運算完成！", state="complete", expanded=False)

    display_cols = ['ID', '股名', '板塊產業', '收盤價', 'MA5 (防守線)', '5MA乖離率(%)', '爆量倍數', vol_label, 'AI 總分']
    st.dataframe(top20[display_cols], use_container_width=True, hide_index=True)
    
    st.info(f"💡 **乖離率實戰指南**：🟢 0% - 3% 首選試單 ｜ 🟡 3% - {user_bias_limit}% 注意追高 ｜ 🔴 >{user_bias_limit}% 已自動扣分處罰。")
    
    # 🔥 升級 5：資金熱區追蹤器
    st.markdown("---")
    st.markdown("### 🔥 今日資金匯聚熱區 (前 20 名板塊統計)")
    sector_counts = top20['板塊產業'].value_counts().reset_index()
    sector_counts.columns = ['板塊產業', '進榜檔數']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sector_counts, hide_index=True, use_container_width=True)
    with col2:
        st.bar_chart(sector_counts.set_index('板塊產業'))

    csv = top20.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下載完整 CSV 報表",
        data=csv,
        file_name=f"Radar_Top20_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
