# Streamlit 仪表盘优化版
# 在Windows上激活虚拟环境：  mykaglightrag_env\Scripts\activate
# 运行： streamlit run hf_dashboard_streamlit.py

import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================
# 1. 数据加载（缓存+加速）
# =========================================
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv(
            "huggingface_models_20251206.csv",
            engine="pyarrow",   # 更快
            dtype={
                "id": "string",
                "author": "string",
                "downloads": "Int64",
                "likes": "Int64",
                "pipeline_tag": "string",
                "trending_score": "Int64",
                "tags": "string",
            }
        )
    except:
        df = pd.read_csv("huggingface_models_20251206.csv")

    # 数据清洗
    df["downloads"] = df["downloads"].fillna(0).astype(int)
    df["likes"] = df["likes"].fillna(0).astype(int)
    df["trending_score"] = df["trending_score"].fillna(0).astype(int)
    df["like_rate"] = df["likes"] / df["downloads"].replace(0, 1)

    return df


df = load_data()

# =========================================
# Streamlit 页面设置
# =========================================
st.set_page_config(page_title="HuggingFace 模型可视化仪表盘", layout="wide")
st.title("📊 HuggingFace 模型可视化仪表盘")

# =========================================
# 顶部指标卡片
# =========================================
col1, col2, col3 = st.columns(3)
col1.metric("模型总数", len(df))
col2.metric("总下载量", int(df["downloads"].sum()))
col3.metric("总点赞量", int(df["likes"].sum()))

# 每个图默认显示前 30，避免全量渲染太慢
TOP_N = 30

# =========================================
# 🔥 下载量 Top 模型
# =========================================
st.subheader("🔥 下载量 Top 模型（前 30）")
top_downloads = df.nlargest(TOP_N, "downloads")
fig1 = px.bar(
    top_downloads,
    x="downloads",
    y="id",
    orientation="h",
)
st.plotly_chart(fig1, use_container_width=True)

# =========================================
# 📈 Trending Score Top
# =========================================
st.subheader("📈 Trending Score Top 模型（前 30）")
top_trend = df.nlargest(TOP_N, "trending_score")
fig4 = px.bar(
    top_trend,
    x="trending_score",
    y="id",
    orientation="h",
)
st.plotly_chart(fig4, use_container_width=True)

# =========================================
# 📈 下载量 vs 点赞量散点图
# =========================================
st.subheader("📈 下载量 vs 点赞量")
fig3 = px.scatter(
    df,
    x="downloads",
    y="likes",
    color="pipeline_tag",
    size="trending_score",
    hover_name="id",
)
st.plotly_chart(fig3, use_container_width=True)

# =========================================
# 📌 Task 模型数量分布
# =========================================
st.subheader("📌 各 Task 模型数量分布")
count_df = df["pipeline_tag"].value_counts().reset_index()
count_df.columns = ["pipeline_tag", "count"]
fig2 = px.pie(count_df, names="pipeline_tag", values="count")
st.plotly_chart(fig2, use_container_width=True)

# =========================================
# 📄 原始数据表
# =========================================
# st.subheader("📄 原始数据表")
# st.dataframe(df)
