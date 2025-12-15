
###
uv venv myenv          # 创建名为 myenv 的虚拟环境
uv venv /path/to/env   # 指定完整路径

myenv\Scripts\activate # 激活虚拟环境
deactivate             # 推出虚拟环境

uv pip install streamlit pandas numpy yfinance plotly

streamlit run app_Closed_Source_Model_Comparative_Analysis.py   #闭源模型分析
streamlit run model_comparison.py                               #闭源模型对比
streamlit run app.py                                            #最近10年汇率走势分析 