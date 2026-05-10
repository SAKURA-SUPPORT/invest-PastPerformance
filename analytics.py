import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from scipy.linalg import eigh
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
import platform

# --- 1. 設定 (Configuration) ---
st.set_page_config(page_title="PCA-SUB Lead-Lag Advisor", layout="wide")

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Meiryo'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'Hiragino Sans'

WINDOW_SIZE = 250
LAMBDA = 0.9

JP_TICKERS = ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T", 
              "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T", 
              "1631.T", "1632.T", "1633.T"]
US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

TICKER_TO_NAME = {
    "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材", "1620.T": "素材・化学",
    "1621.T": "医薬品", "1622.T": "自動車・輸送機", "1623.T": "鉄鋼・非鉄", "1624.T": "機械",
    "1625.T": "電機・精密", "1626.T": "情報通信・サービス", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
    "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行", "1632.T": "金融（除く銀行）", "1633.T": "不動産"
}

# --- 2. 理論コア (Core Logic) ---
def build_C0(nu, nj, Ct):
    C0 = np.zeros((nu + nj, nu + nj))
    C0[:nu, :nu] = Ct[:nu, :nu]
    C0[nu:, nu:] = Ct[nu:, nu:]
    return C0

def get_latest_complete_data():
    all_tickers = US_TICKERS + JP_TICKERS
    data = yf.download(all_tickers, start=datetime.now() - timedelta(days=450), multi_level_index=False)
    
    # 米国: Close-to-Close リターン
    us_close = data['Close'][US_TICKERS].ffill()
    us_ret = us_close.pct_change().dropna()
    
    # 日本: Open-to-Close リターン (実績値用)
    jp_open = data['Open'][JP_TICKERS]
    jp_close = data['Close'][JP_TICKERS]
    jp_otc_ret = (jp_close / jp_open - 1).dropna()
    
    # シグナル日(t)の翌営業日が日本(t+1)に存在し、かつJP実績が確定している日を探す
    jpx = mcal.get_calendar('JPX')
    valid_us_dates = us_ret.index
    
    target_sig_date = None
    target_pred_date = None
    
    for sig_date in reversed(valid_us_dates):
        # sig_date の翌営業日を取得
        next_days = jpx.valid_days(start_date=sig_date + timedelta(days=1), end_date=sig_date + timedelta(days=10))
        pred_date = next_days[0].replace(tzinfo=None)
        
        if pred_date in jp_otc_ret.index:
            target_sig_date = sig_date
            target_pred_date = pred_date
            break
            
    if not target_sig_date:
        return None

    # PCA-SUB 計算 (sig_date 時点までの WINDOW_SIZE を使用)
    us_train = us_ret.loc[:target_sig_date].tail(WINDOW_SIZE)
    # 相関計算用には JP の Close-to-Close を使用 (analytics.py 準拠)
    jp_ctc_ret = jp_close.pct_change().loc[:target_sig_date].tail(WINDOW_SIZE)
    
    common_dates = us_train.index.intersection(jp_ctc_ret.index)
    z_u = (us_train.loc[common_dates] - us_train.loc[common_dates].mean()) / us_train.loc[common_dates].std()
    z_j = (jp_ctc_ret.loc[common_dates] - jp_ctc_ret.loc[common_dates].mean()) / jp_ctc_ret.loc[common_dates].std()
    
    Ct = pd.concat([z_u, z_j], axis=1).corr().values
    nu, nj = len(US_TICKERS), len(JP_TICKERS)
    Ct_reg = (1 - LAMBDA) * Ct + LAMBDA * build_C0(nu, nj, Ct)
    
    evals, evecs = eigh(Ct_reg, subset_by_index=[(nu+nj)-3, (nu+nj)-1])
    B = evecs[nu:, :] @ evecs[:nu, :].T
    
    # シグナルと実績
    latest_shock = z_u.iloc[-1].values
    scores = B @ latest_shock
    actuals = jp_otc_ret.loc[target_pred_date]
    
    return scores, actuals, target_sig_date, target_pred_date

# --- 3. UI & 実行 ---
st.title("日米業種リードラグ・実績同期アドバイザー")

if st.button("実績確定済みの最新データを抽出 & シート更新"):
    result = get_latest_complete_data()
    
    if result:
        scores, actuals, sig_date, pred_date = result
        
        df_res = pd.DataFrame({
            'Ticker': JP_TICKERS,
            'Name': [TICKER_TO_NAME[t] for t in JP_TICKERS],
            'Score': scores,
            'Actual': [actuals[t] for t in JP_TICKERS]
        }).sort_values('Score', ascending=False)

        st.write(f"### 抽出対象: {pred_date.strftime('%Y/%m/%d')} (実績確定済み)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_res.style.format({'Score': '{:.3f}', 'Actual': '{:.2%}'})
                         .background_gradient(cmap='RdYlGn', subset=['Score']))
        with col2:
            fig, ax = plt.subplots()
            ax.barh(df_res['Name'], df_res['Score'], color=['#2ecc71' if x > 0 else '#e74c3c' for x in df_res['Score']])
            ax.invert_yaxis()
            st.pyplot(fig)

        # スプレッドシート出力
        try:
            creds_dict = st.secrets["gcp_service_account"]
            client = gspread.authorize(Credentials.from_service_account_info(creds_dict, 
                                        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']))
            sheet = client.open("過去実績分析").get_worksheet(0)
            
            buy_top = df_res.head(3)
            sell_bot = df_res.tail(3).iloc[::-1]
            
            row = [sig_date.strftime('%Y/%m/%d'), pred_date.strftime('%Y/%m/%d')]
            for _, r in buy_top.iterrows():
                row.extend([r['Name'], round(r['Score'], 3), f"{r['Actual']:.2%}"])
            for _, r in sell_bot.iterrows():
                row.extend([r['Name'], round(r['Score'], 3), f"{r['Actual']:.2%}"])
            
            sheet.insert_row(row, 2)
            st.success(f"{pred_date.strftime('%Y/%m/%d')} の実績データを2行目に挿入しました。")
            
        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.warning("実績が確定した新しいデータが見つかりませんでした。")