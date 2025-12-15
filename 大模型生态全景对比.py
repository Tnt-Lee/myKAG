import streamlit as st
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="2025 大模型生态全景对比",
    page_icon="🌐",
    layout="wide"
)

# 标题
st.title("🌐 2025 年 12 月大模型生态全景对比")

# 数据
data = {
    "模型体系": ["Alibaba Qwen", "Anthropic Claude", "OpenAI", "Google AI", "Baidu ERNIE", "Cohere", "Huawei Pangu", "Meta AI", "Microsoft AI", "Stability AI", "Perplexity AI", "Tencent HY"],
    "核心定位": [
        "开源筑生态 + 闭源打高端，覆盖多领域",
        "企业级高性能推理，注重安全对齐",
        "通用 AI 领导者，多模态与推理",
        "多模态创新，具身智能",
        "企业级应用，智能体优先",
        "企业级定制化，高性价比",
        "行业 AI 解决方案，全栈自研",
        "开源权重 + 闭源服务，多模态",
        "企业级 AI 平台，模型无关",
        "多模态生成，开源权重",
        "检索增强，知识溯源",
        "全链路自研，多模态覆盖"
    ],
    "旗舰模型": [
        "Qwen3-Max（>1 万亿参数 MoE，1M tokens 上下文）",
        "Claude 3.5 Sonnet（200K tokens 上下文，多模态）",
        "GPT-5.1（400K tokens 上下文，推理旗舰）",
        "Gemini 3 Pro（1M tokens 上下文，多模态推理旗舰）",
        "文心大模型 4.0 Turbo（32K–128K 上下文，多模态）",
        "Command A Reasoning（128K–256K tokens 上下文）",
        "盘古大模型 5.5（支持智能驾驶世界模型）",
        "Llama 4（MoE 架构，128K 上下文）",
        "GPT-5.1（推理旗舰）",
        "Stable Diffusion XL（图像生成）",
        "Sonar Online（检索增强生成）",
        "Tencent HY 2.0 Think（128K 输入，64K 输出）"
    ],
    "多模态能力": [
        "部分闭源模型支持，如 Qwen3-Max-Thinking-Heavy",
        "图像输入，未来可能支持更多",
        "图像输入（GPT-5 系列），音频输出（GPT-4o）",
        "全模态支持（文本、图像、音频、视频）",
        "文本 + 图像",
        "文本 + 图像",
        "文本 + 图像",
        "无（Llama 系列）",
        "多模态（GPT 系列）",
        "图像生成（Stable Diffusion）",
        "无",
        "无"
    ],
    "推理性能": [
        "Qwen3-Max 性能全球前三，投资模拟赛收益率 22.32% 夺冠",
        "Claude 3.5 Sonnet 编码、推理、视觉理解全面超越 GPT-4o",
        "GPT-5.1 强推理能力，支持 reasoning effort 配置",
        "Gemini 3 Pro 自适应思考，LMArena 排行第一",
        "文心大模型 4.0 Turbo 推理速度提升 230%",
        "Command A Reasoning 企业级推理优化",
        "盘古大模型 5.5 智能驾驶世界模型",
        "Llama 4 性能未达预期",
        "GPT-5.1 推理旗舰",
        "Stable Diffusion XL 图像生成性能优化",
        "Sonar Online 检索增强，引用丰富",
        "Tencent HY 2.0 Think 强逻辑推理"
    ],
    "成本效率": [
        "闭源模型通过阿里云百炼 API 提供，成本未知",
        "按需报价，Sonnet 性价比高",
        "成本未知，GPT-5 系列价格较高",
        "成本未知，Gemini 3 Pro 高性能",
        "成本未知，文心大模型 4.0 Turbo 高性价比",
        "Command R7B 最便宜，适合高频调用",
        "盘古大模型 5.5 企业级成本未知",
        "Llama 4 部署成本高",
        "成本未知，GPT-5.1 高性能",
        "Stable Diffusion XL 开源权重，社区支持",
        "Sonar Online 免费试用，Pro 订阅收费",
        "Tencent HY 2.0 Think 成本未知"
    ],
    "安全对齐": [
        "未明确提及",
        "安全对齐特性完善，如 Constitutional AI",
        "未明确提及",
        "未明确提及",
        "未明确提及",
        "企业级安全，支持私有化部署",
        "未明确提及",
        "未明确提及",
        "未明确提及",
        "未明确提及",
        "Sonar 强制事实绑定，引用可点击",
        "未明确提及"
    ],
    "企业应用": [
        "通过阿里云百炼提供企业级 API",
        "企业级主力模型，支持私有化部署",
        "企业级应用广泛，通过 Azure OpenAI Service 提供",
        "企业级应用，如智能驾驶、机器人控制",
        "企业级应用，如智能客服、数据分析",
        "企业级应用，如智能客服、知识管理",
        "企业级应用，如矿山、金融、气象",
        "企业级应用，如广告、品牌音效",
        "企业级应用，如智能客服、知识管理",
        "企业级应用，如广告、品牌音效",
        "企业级应用，如智能客服、知识管理",
        "企业级应用，如智能客服、知识管理"
    ],
    "开源生态": [
        "开源模型家族丰富，如 Qwen3 系列",
        "无开源模型",
        "无开源模型",
        "无开源模型",
        "无开源模型",
        "无开源模型",
        "开源轻量模型，如 Pangu-7B",
        "开源权重模型，如 Llama 4",
        "无开源模型",
        "开源权重模型，如 Stable Diffusion XL",
        "无开源模型",
        "无开源模型"
    ],
    "访问方式": [
        "Qwen Chat、阿里云百炼 API",
        "Anthropic API、Web 应用、企业方案",
        "Azure OpenAI Service",
        "Vertex AI Model Garden",
        "百度智能云千帆、文心智能体平台",
        "Cohere Platform、Hugging Face、Azure Marketplace",
        "ModelArts Studio",
        "Hugging Face、Meta AI 官方 API",
        "Azure OpenAI Studio",
        "Stability AI Platform、Hugging Face",
        "Perplexity 官网、Comet AI 浏览器",
        "腾讯云控制台、元宝 APP"
    ]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 选择显示的列
st.sidebar.title("选择显示的列")
columns = df.columns.tolist()
selected_columns = st.sidebar.multiselect("选择列", columns, default=columns)

# 显示表格
st.subheader("大模型生态全景对比")
st.table(df[selected_columns])

# 页脚
st.divider()
st.caption("数据截至 2025 年 12 月 | 来源：各公司官网、文档及公开资料整理")