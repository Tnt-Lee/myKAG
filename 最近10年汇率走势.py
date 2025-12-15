import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="汇率走势分析", layout="wide")

st.title("📊 最近10年汇率走势分析")
st.markdown("---")


def safe_download_close(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"Failed to download {ticker}")
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze(axis=1)  # 转为 Series
    return close

# 获取数据
@st.cache_data
def get_exchange_rates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    usd_jpy = safe_download_close('JPY=X', start_date, end_date)
    usd_cny = safe_download_close('CNY=X', start_date, end_date)
    # ⭐ 关键：按共同交易日对齐
    usd_jpy, usd_cny = usd_jpy.align(usd_cny, join='inner')
    jpy_cny = usd_cny / usd_jpy
    jpy_usd = usd_jpy   # ← ⭐ 新增这一行

    # 现在可以安全使用 to_frame()
    df_result = pd.DataFrame({
        'JPY/CNY': jpy_cny,
        'USD/CNY': usd_cny,
        'USD/JPY': jpy_usd    # ← ⭐ 新增这一列
    })
    return df_result.dropna()

# 加载数据
with st.spinner('正在加载汇率数据...'):
    df = get_exchange_rates()

if df is not None:
    st.success('✅ 数据加载完成')
    
    # 显示数据摘要
    col1, col2, col3 = st.columns(3)
    latest_date = df.index.max().date()
    
    with col1:
        st.metric(
            "日元/人民币",
            f"{df['JPY/CNY'].iloc[-1]:.4f}",
            f"{df['JPY/CNY'].iloc[-1] - df['JPY/CNY'].iloc[0]:.4f}"
        )
        st.caption(f"数据日期：{latest_date}")
    
    with col2:
        st.metric(
            "美元/人民币",
            f"{df['USD/CNY'].iloc[-1]:.4f}",
            f"{df['USD/CNY'].iloc[-1] - df['USD/CNY'].iloc[0]:.4f}"
        )
        st.caption(f"数据日期：{latest_date}")
    
    with col3:
        st.metric(
            "美元/日元",
            f"{df['USD/JPY'].iloc[-1]:.4f}",
            f"{df['USD/JPY'].iloc[-1] - df['USD/JPY'].iloc[0]:.4f}"
        )
        st.caption(f"数据日期：{latest_date}")
    
    st.markdown("---")
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("日元对人民币 (JPY/CNY)", "美元对人民币 (USD/CNY)", "日元对美元 (USD/JPY)"),
        vertical_spacing=0.12
    )
    
    # 日元对人民币
    fig.add_trace(
        go.Scatter(x=df.index, y=df['JPY/CNY'], name='JPY/CNY',
                   line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )
    
    # 美元对人民币
    fig.add_trace(
        go.Scatter(x=df.index, y=df['USD/CNY'], name='USD/CNY',
                   line=dict(color='#4ECDC4', width=2)),
        row=2, col=1
    )
    
    # 日元对美元
    fig.add_trace(
        go.Scatter(x=df.index, y=df['USD/JPY'], name='USD/JPY',
                   line=dict(color='#FFE66D', width=2)),
        row=3, col=1
    )
    
    fig.update_yaxes(title_text="汇率", row=1, col=1)
    fig.update_yaxes(title_text="汇率", row=2, col=1)
    fig.update_yaxes(title_text="汇率", row=3, col=1)
    
    fig.update_xaxes(title_text="日期", row=3, col=1)
    
    fig.update_layout(height=1000, hovermode='x unified', template='plotly_white')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 对比分析")
    
    # 计算统计数据
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**日元/人民币**")
        st.write(f"最高: {df['JPY/CNY'].max():.4f}")
        st.write(f"最低: {df['JPY/CNY'].min():.4f}")
        st.write(f"平均: {df['JPY/CNY'].mean():.4f}")
        st.write(f"变化幅度: {((df['JPY/CNY'].max() - df['JPY/CNY'].min()) / df['JPY/CNY'].min() * 100):.2f}%")
    
    with col2:
        st.write("**美元/人民币**")
        st.write(f"最高: {df['USD/CNY'].max():.4f}")
        st.write(f"最低: {df['USD/CNY'].min():.4f}")
        st.write(f"平均: {df['USD/CNY'].mean():.4f}")
        st.write(f"变化幅度: {((df['USD/CNY'].max() - df['USD/CNY'].min()) / df['USD/CNY'].min() * 100):.2f}%")
    
    with col3:
        st.write("**日元/美元**")
        st.write(f"最高: {df['USD/JPY'].max():.4f}")
        st.write(f"最低: {df['USD/JPY'].min():.4f}")
        st.write(f"平均: {df['USD/JPY'].mean():.4f}")
        st.write(f"变化幅度: {((df['USD/JPY'].max() - df['USD/JPY'].min()) / df['USD/JPY'].min() * 100):.2f}%")
    
    st.markdown("---")
    st.subheader("💡 趋势分析")
    
    analysis = f"""
    **关键发现：**
    
    1. **美元/人民币走势**: 过去10年美元对人民币整体呈贬值趋势（数值越大表示人民币相对贬值）。
    
    2. **日元/人民币走势**: 日元对人民币的波动相对较小，保持在较稳定的区间。
    
    3. **日元/美元走势**: 显示美元相对日元有升值的总体趋势，这反映了美元在国际市场上的强势。
    
    4. **相关性**: 美元强势时，美元/人民币和日元/美元都会上升，显示美元对其他货币的压制效应。
    
    5. **投资启示**: 这三个汇率对进出口贸易、外汇交易和跨国投资都有重要影响。
    """
    
    st.markdown(analysis)
    
    st.markdown("---")
    st.subheader("📊 原始数据")
    
    if st.checkbox("显示原始数据"):
        st.dataframe(df.tail(100), use_container_width=True)

else:
    st.error("无法加载数据，请检查网络连接")