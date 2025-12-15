import streamlit as st
import pandas as pd

# 页面配置 streamlit run app_Closed_Source_Model_Comparative_Analysis.py
st.set_page_config(
    page_title="2025 大模型生态全景",
    page_icon="🤖",
)

st.title("🌐 2025 年 12 月大模型生态全景对比")
st.caption("涵盖 Alibaba Qwen、Anthropic Claude、OpenAI 官方模型体系")

# ======================
# 阿里巴巴 Qwen
# ======================

with st.expander("📌 战略概览：开源筑生态 + 闭源打高端", expanded=False):
    st.markdown("""
    - **开源侧**：Hugging Face / ModelScope 开放数十款模型（Apache 2.0），全球下载 **6 亿+**，衍生模型 **17 万+**
    - **闭源侧**：通过 **阿里云百炼** 和 **Qwen Chat** 提供 API（如 Qwen3-Max），不开放权重
    """)

tab_cn, tab_global = st.tabs(["🇨🇳 国内模型", "🌍 海外模型"])

with tab_cn:
    tab_qwen, tab_baidu, tab_huawei, tab_tencent = st.tabs([
        "🧠 Qwen", "🪞 百度", "PG 华为", "HY 腾讯"
    ])


with tab_global:
    tab_claude, tab_openai, tab_google_ai, tab_cohere, tab_deepmind, tab_meta, tab_microsoft,tab_stability, tab_perplexity = st.tabs([
        "⚖️ Anthropic Claude",
        "🔵 OpenAI",
        "🟣 Google AI",
        "🇨🇦 Cohere",
        "🔬 Google DeepMind",
        "🦛 Meta AI",
        "🔷 Microsoft AI",
        "🎨 Stability AI",
        "🔍 Perplexity AI",
    ])

# ----------------------
# Qwen Tab
# ----------------------
with tab_qwen:
    st.header("🧠 一、通义千问（Qwen）主干语言模型")

    st.subheader("✅ 1. 闭源旗舰模型（仅限 API / Qwen Chat）")
    qwen_closed = pd.DataFrame({
        "模型名称": ["Qwen3-Max", "Qwen3-Max-Thinking-Heavy"],
        "发布时间": ["2025 年 9 月", "2025 年 11 月"],
        "参数规模": [">1 万亿（MoE）", "同上"],
        "上下文": ["**1M tokens**", "1M+"],
        "核心能力": [
            "- 全球前三（LMArena）\n- SWE-Bench Verified **69.6 分**\n- Tau2-Bench **74.8 分**\n- AIME 25 / HMMT 数学推理 **满分 100**",
            "增强并行推理 + 代码解释器，专攻高难度数学/科研任务"
        ],
        "访问方式": [
            "阿里云百炼 API\nQwen Chat（免费体验）",
            "百炼平台（企业版）"
        ]
    })
    st.dataframe(qwen_closed, use_container_width=True, hide_index=True)

    st.info("**Qwen3-Max 是阿里当前最强模型**，性能超越 GPT-5、Claude Opus4，在 **投资模拟赛中收益率达 22.32% 夺冠**。")

    st.subheader("✅ 2. 开源模型家族（可本地部署）")
    st.markdown("#### 🔹 Qwen3 系列（2025 年 4 月发布）")
    qwen3_open = pd.DataFrame({
        "模型": ["Qwen3-235B-A22B", "Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-0.6B ~ 14B"],
        "架构": ["MoE", "MoE", "稠密", "稠密"],
        "参数": ["总 235B / 激活 22B", "总 30B / 激活 3B", "32B", "多档"],
        "上下文": ["32K", "32K", "32K", "32K"],
        "特点": [
            "开源最大 MoE 模型，性能接近 Qwen3-Max",
            "可在 **普通 CPU + 32GB 内存** 运行",
            "强力稠密模型，适合 GPU 部署",
            "覆盖从手机到服务器全场景"
        ]
    })
    st.dataframe(qwen3_open, use_container_width=True, hide_index=True)

    st.markdown("#### 🔹 Qwen2.5 系列（成熟稳定）")
    qwen25_open = pd.DataFrame({
        "规格": ["Qwen2.5-0.5B ~ 72B"],
        "参数": ["稠密"],
        "上下文": ["**128K**"],
        "训练数据": ["18T tokens"],
        "推荐用途": ["个人学习首选 **7B**（RTX 3060 可跑）"]
    })
    st.dataframe(qwen25_open, use_container_width=True, hide_index=True)

    st.markdown("📥 所有开源模型可在 [Hugging Face](https://huggingface.co/Qwen) 或 [魔搭 ModelScope](https://modelscope.cn/models/qwen) 下载。")

    st.header("🎨 二、多模态与专用模型")
    multi_models = pd.DataFrame({
        "模型系列": ["Qwen-VL / Qwen2-VL", "Qwen-Audio", "Qwen-Image-Edit", "Qwen3-TTS", "Qwen-Coder", "Qwen-Math", "QwQ / QVQ"],
        "类型": ["视觉-语言", "音频-语言", "图像编辑", "语音合成", "代码模型", "数学推理", "推理思考"],
        "能力": [
            "图像理解、OCR、图表问答",
            "语音识别、音频事件理解",
            "语义+外观联合编辑（2025 年 8 月）",
            "**49 种音色**、10 语种 + 方言",
            "支持 80+ 编程语言，SWE-Bench 高分",
            "专精奥数、高数、符号计算",
            "“慢思考”模式，擅长复杂推理链"
        ],
        "开源状态": ["✅ 开源", "✅ 开源", "❌ 闭源（API 服务）", "❌ 闭源（Qwen API）", "✅ 开源", "✅ 开源", "✅ 开源"]
    })
    st.dataframe(multi_models, use_container_width=True, hide_index=True)

    st.header("🔍 三、嵌入与向量模型")
    embed_models = pd.DataFrame({
        "模型": ["Qwen3-Embedding", "Qwen-Embedding-v2"],
        "输入长度": ["8K tokens", "512 tokens"],
        "输出维度": ["1024", "768"],
        "性能提升": ["文本检索 **+40%** vs 上一代", "轻量级通用嵌入"]
    })
    st.dataframe(embed_models, use_container_width=True, hide_index=True)

    st.header("🤖 四、智能体与行业模型")
    agent_models = pd.DataFrame({
        "模型/平台": ["Qwen-Agent", "东方航空行程规划 Agent", "Qwen-SEA-LION-v4", "千问 App"],
        "应用场景": ["智能体开发框架", "航旅", "东南亚语言", "个人 AI 助手"],
        "案例": [
            "支持 MCP 协议、工具调用、记忆管理",
            "2025 年 12 月上线，支持航班组合、时差换算、转机衔接",
            "新加坡国家 AI 计划采用，支持印尼语、泰语等",
            "支持 **119 种语言实时翻译**、图像识别、语音交互，接入支付宝/高德生态"
        ]
    })
    st.dataframe(agent_models, use_container_width=True, hide_index=True)

    st.header("⚙️ 五、底层技术与部署")
    st.markdown("""
    - **训练框架**：PAI-FlashMoE（MoE 训练效率 +30%）
    - **长序列优化**：ChunkFlow（1M token 训练吞吐 +3x）
    - **推理加速**：vLLM、TensorRT-LLM、Ollama（`ollama run qwen2.5:7b`）
    - **部署平台**：
        - **Ollama / LM Studio**：本地快速体验  
        - **阿里云百炼**：企业级 API / 微调 / 私有化
    """)

    st.header("🌐 访问方式汇总")
    access_qwen = pd.DataFrame({
        "类型": ["闭源模型体验", "API 调用", "开源模型下载", "本地运行"],
        "平台": ["Qwen Chat", "阿里云百炼", "ModelScope（魔搭）", "Ollama"],
        "地址": [
            "[https://chat.qwen.ai](https://chat.qwen.ai/)",
            "[https://bailian.console.aliyun.com](https://www.qianwen.com/chat/https)",
            "[https://modelscope.cn/models/qwen](https://modelscope.cn/models/qwen)",
            "`ollama run qwen3:32b`"
        ]
    })
    st.dataframe(access_qwen, use_container_width=True, hide_index=True)

    st.header("✅ 总结：阿里巴巴模型战略")
    st.table(pd.DataFrame({
        "维度": ["开源", "闭源", "多模态", "落地"],
        "策略": [
            "全尺寸覆盖（0.5B–235B），Apache 2.0 商用友好，构建全球最大中文开源生态",
            "Qwen3-Max 等旗舰模型对标 GPT-5/Claude Opus，聚焦企业高价值场景",
            "文本、图像、音频、视频、代码、数学全覆盖",
            "深度集成至 **千问 App、支付宝、高德、夸克、航司、政务** 等场景"
        ]
    }))

    st.success("""
    **推荐选择**：
    - **个人学习** → `Qwen2.5-7B`（平衡性能与硬件）
    - **企业应用** → `Qwen3-Max`（API）或 `Qwen3-30B-A3B`（本地部署）
    - **编程任务** → `Qwen-Coder` 或 `Qwen3-Max`
    """)

# ----------------------
# Claude Tab
# ----------------------
with tab_claude:
    st.header("🧠 Anthropic Claude 模型全系列表（2025 年最新）")

    st.subheader("✅ 一、Claude 3.5 系列（2024–2025 年发布｜当前主力）")
    claude_35 = pd.DataFrame({
        "模型名称": ["Claude 3.5 Sonnet", "Claude 3.5 Haiku", "Claude 3.5 Opus"],
        "发布时间": ["2024 年 6 月", "2025 年初（预览）", "尚未正式发布（传闻中）"],
        "上下文窗口": ["**200K tokens**", "200K tokens", "预计 200K+"],
        "多模态": ["✅（图像输入）", "✅（图像输入）", "预计支持"],
        "核心优势": [
            "**最强综合性能**：编码、推理、视觉理解全面超越 Opus；速度比 Opus 快 2 倍，成本更低",
            "超快响应、极低成本，接近实时交互",
            "可能为未来旗舰，但目前 **Sonnet 已超越原 Opus**"
        ],
        "适用场景": ["开发者首选、智能体、复杂任务", "聊天机器人、高频 API 调用", "—"]
    })
    st.dataframe(claude_35, use_container_width=True, hide_index=True)

    st.info("""
    **重要说明**：
    - **Claude 3.5 Sonnet 是目前 Anthropic 最强公开模型**，在 HumanEval（代码）、MMLU（知识）、DROP（推理）等基准上全面领先 GPT-4o 和 Claude 3 Opus。
    - 官方已明确：**Sonnet 在多数任务上优于 Opus，且更便宜更快**，因此推荐优先使用 Sonnet。
    """)

    st.subheader("✅ 二、Claude 3 系列（2024 年 3 月发布｜逐步被 3.5 取代）")
    claude_3 = pd.DataFrame({
        "模型名称": ["Claude 3 Opus", "Claude 3 Sonnet", "Claude 3 Haiku"],
        "上下文窗口": ["200K tokens"] * 3,
        "多模态": ["✅"] * 3,
        "特点": [
            "最强推理、创意写作、战略分析",
            "平衡速度/成本/能力，企业级主力",
            "最快、最便宜，适合简单任务"
        ],
        "当前状态": [
            "仍可用，但性能已被 3.5 Sonnet 超越",
            "逐步被 3.5 Sonnet 替代",
            "仍广泛用于轻量场景"
        ]
    })
    st.dataframe(claude_3, use_container_width=True, hide_index=True)

    st.warning("注意：Claude 3 系列所有模型均支持 **图像输入**（如截图、图表、PDF 页面），但 **不支持图像生成或音频/视频**。")

    st.header("🔒 三、安全与对齐特性（所有 Claude 模型通用）")
    st.markdown("""
    - **Constitutional AI**：通过自我批评和宪法约束训练，减少有害输出。
    - **工具使用（Tool Use）**：支持函数调用（JSON Schema），可集成外部 API。
    - **结构化输出（JSON Mode）**：强制模型返回有效 JSON，便于程序解析。
    - **缓存（Prompt Caching）**：重复提示可大幅降低延迟和成本（仅部分模型支持）。
    """)

    st.header("📈 四、性能与定价参考（以 Claude 3.5 Sonnet 为例）")
    pricing = pd.DataFrame({
        "项目": ["输入", "输出", "缓存输入"],
        "价格（每百万 tokens）": ["**$3.00**", "**$15.00**", "**$0.30**"]
    })
    st.dataframe(pricing, use_container_width=True, hide_index=True)
    st.markdown("> 💡 相比 Claude 3 Opus（输入 $15 / 输出 $75），**3.5 Sonnet 性价比提升 5 倍以上**。")

    st.header("🌐 五、访问方式")
    st.markdown("""
    - **API**：通过 [Anthropic API](https://docs.anthropic.com/claude) 调用（需 API Key）
    - **Web 应用**：[Claude.ai](https://claude.ai/)（支持文件上传、多轮对话）
    - **企业方案**：Claude for Enterprise（支持 SSO、审计日志、私有部署选项）
    """)

    st.header("❌ 已弃用/不存在的模型")
    st.markdown("""
    - **Claude 2.1 / 2.0**：仍可调用，但强烈建议升级至 Claude 3.5。
    - **Claude Instant**：旧版轻量模型，已被 Haiku 取代。
    - **Claude 4**：**尚未发布**，官方未确认命名；当前最新为 **Claude 3.5**。
    """)

    st.header("✅ 总结：推荐使用策略")
    rec_claude = pd.DataFrame({
        "需求": [
            "最强性能（编码/推理/多模态）",
            "超低延迟、高吞吐",
            "企业级稳定部署",
            "简单问答/聊天"
        ],
        "推荐模型": [
            "**Claude 3.5 Sonnet** ✅（首选）",
            "Claude 3.5 Haiku（若可用）或 Claude 3 Haiku",
            "Claude 3.5 Sonnet + Prompt Caching",
            "Claude 3 Haiku"
        ]
    })
    st.dataframe(rec_claude, use_container_width=True, hide_index=True)

    st.info("Anthropic 的核心理念是 **“Helpful, Harmless, Honest”**，所有模型均经过严格安全对齐，适合对伦理和可靠性要求高的场景。")

# ----------------------
# OpenAI Tab
# ----------------------
with tab_openai:
    st.header("🔵 OpenAI 主流闭源大语言模型对比表（2025年12月）")

    openai_models = pd.DataFrame({
        "模型名称": ["GPT-5.1", "GPT-5 Pro", "GPT-5", "GPT-5 Mini", "GPT-5 Nano", "GPT-4.1", "GPT-4o", "GPT-4o Mini"],
        "类型": ["Frontier"] * 5 + ["Non-reasoning", "Multimodal", "Multimodal"],
        "上下文长度": ["400K"] * 5 + ["~1M", "128K", "128K"],
        "输入模态": ["文本、图像"] * 8,
        "输出模态": ["文本"] * 6 + ["文本、音频"] * 2,
        "输入价格（每百万 token）": ["$1.25", "$15.00", "$1.25", "$0.25", "$0.05", "$2.00", "$2.50", "$0.15"],
        "输出价格（每百万 token）": ["$10.00", "$120.00", "$10.00", "$2.00", "$0.40", "$8.00", "$10.00", "$0.60"],
        "典型用途": [
            "编程、Agent、复杂推理",
            "高难度科研、深度推理",
            "通用高级任务（旧版）",
            "快速响应、结构化任务",
            "分类、摘要、轻量任务",
            "工具调用、指令遵循",
            "实时交互、多模态",
            "轻量多模态应用"
        ]
    })
    st.dataframe(openai_models, use_container_width=True, hide_index=True)

    st.markdown("""
    > 💡 注：
    > - “推理能力”指模型是否具备显式推理步骤（如 o1/GPT-5 系列的 reasoning tokens）。
    > - GPT-5 系列为当前主力模型线，替代了旧 o1/o3 系列。
    > - GPT-4.1 是**非推理型**但高智能模型，适合低延迟场景。
    > - GPT-4o 及 Mini 支持**实时语音输入/输出**，适用于语音助手等场景。
    """)

    st.header("🧠 专用模型（Specialized Models）")
    special_models = pd.DataFrame({
        "模型名称": [
            "Sora 2", "Sora 2 Pro", "GPT Image 1", "GPT Image 1 Mini",
            "GPT-4o Transcribe", "GPT-4o Mini TTS", "o3-deep-research", "o4-mini-deep-research"
        ],
        "类型": ["视频生成"] * 2 + ["图像生成"] * 2 + ["语音识别", "语音合成", "研究专用", "研究专用"],
        "功能": [
            "带同步音频的视频生成", "高质量长视频生成", "文生图", "低成本文生图",
            "语音转文字（带说话人分离）", "文字转语音", "深度文献分析、推理", "快速研究辅助"
        ],
        "价格模型": ["按生成计费", "高于 Sora 2", "按图像计费", "低于 GPT Image 1", "按分钟计费", "按字符计费", "高单价", "中等"]
    })
    st.dataframe(special_models, use_container_width=True, hide_index=True)

    st.header("🔓 开源权重模型（Open-weight under Apache 2.0）")
    oss_models = pd.DataFrame({
        "模型": ["gpt-oss-120b", "gpt-oss-20b"],
        "参数规模": ["~120B", "~20B"],
        "特点": ["可在单张 H100 运行", "低延迟、小资源占用"],
        "适用场景": ["本地部署、研究", "边缘设备、快速推理"]
    })
    st.dataframe(oss_models, use_container_width=True, hide_index=True)

    st.header("📌 模型演进路线简析")
    st.markdown("""
    - **推理能力演进**：  
        `o1 → o3 → GPT-5 → GPT-5.1`  
        新模型支持 **reasoning effort 配置**（高/低/关闭），实现性能与成本平衡。
        
    - **多模态演进**：  
        `GPT-4o → GPT-4o Mini → GPT-5 系列（图像输入）`  
        当前 GPT-5 系列暂不支持**音频/视频输出**，多模态以 GPT-4o 为主力。
        
    - **成本优化策略**：  
        OpenAI 推出 **Nano / Mini / Pro** 三级分层，覆盖从 $0.05 到 $120/M token 的全场景需求。
    """)

    st.header("❌ 已废弃或不推荐用于 API 的模型")
    st.markdown("""
    - DALL·E 2 / DALL·E 3（被 GPT Image 1 取代）
    - GPT-4 Turbo / GPT-4（被 GPT-4o 和 GPT-5 系列取代）
    - o1 / o1-mini / o1-preview（被 GPT-5 系列取代）
    - ChatGPT 专属模型（如 `gpt-5-chat-latest`）——**不建议用于生产 API**
    """)
# ----------------------
# 百度文心 Tab
# ----------------------
with tab_baidu:
    st.header("🪞 百度文心大模型（ERNIE Bot）全栈体系")

    st.markdown("""
    > 所有模型均基于 **飞桨（PaddlePaddle）框架** 开发，支持通过 **百度智能云千帆大模型平台** 或 **文心智能体平台** 调用。
    """)

    st.subheader("✅ 一、文心大模型（ERNIE Bot）主干系列")

    ernie_main = pd.DataFrame({
        "模型名称": [
            "文心大模型 4.0 Turbo",
            "文心大模型 4.0",
            "文心大模型 4.0 工具版",
            "文心大模型 3.5"
        ],
        "发布时间": ["2025 年 8 月", "2024 年底", "2024 年 11 月", "2023 年中"],
        "上下文窗口": ["32K–128K", "128K", "-", "8K–32K"],
        "多模态": ["✅（文本+图像）"] * 4,
        "核心特点": [
            "推理速度提升 **230%**（达 396 tokens/s），中文理解 CLUE 91.7 分",
            "混合专家架构（MoE），万亿级参数，支持动态注意力与多模态统一表征",
            "强化 **代码生成、数据处理、图表输出** 能力",
            "稳定成熟，广泛用于企业服务"
        ],
        "定位": [
            "**高性能推理首选**",
            "旗舰通用模型",
            "数据分析 & 决策支持",
            "成本敏感型场景"
        ]
    })
    st.dataframe(ernie_main, use_container_width=True, hide_index=True)

    st.info("""
    **关键升级**：
    - **4.0 Turbo** 在保持效果的同时大幅提速降本，适合高并发业务。
    - **4.0 标准版** 更适合复杂任务（如长文档理解、多轮规划）。
    """)

    st.subheader("🤖 二、文心智能体（Agent）专用模型")
    st.markdown("> 基于文心 4.0 构建，支持 **理解 → 规划 → 反思 → 进化** 四步智能体框架。")

    agents = pd.DataFrame({
        "智能体名称": ["文小言", "农民院士智能体", "上体体育大模型", "金融智能体矩阵", "城市大模型底座"],
        "领域": ["通用助手", "农业", "体育", "金融", "政务/城市治理"],
        "功能": [
            "原“文心一言”升级版，搭载 4.0 模型，支持多模态交互、个性化陪伴",
            "联合朱有勇院士团队打造，解答种植、病虫害等问题",
            "服务国家队（游泳、田径、体操等），提供训练分析与战术建议",
            "风控、投研、客服自动化",
            "整合交通、应急、民生问答"
        ]
    })
    st.dataframe(agents, use_container_width=True, hide_index=True)

    st.success("✅ 所有智能体可在 **[文心智能体平台](https://agents.baidu.com/)** 免费创建和部署。")

    st.subheader("💻 三、代码与开发工具模型")
    st.markdown("**文心快码**（Wenxin CodeGen）| 智能代码助手")
    st.markdown("""
    - 百度内部：代码提交量 +35%，研发提效 14%  
    - 支持 Java/Python/C++ 等主流语言  
    - 已服务 **超 1 万家企业**（金融、汽车、制造）
    """)
    st.markdown("> 基于 **CodeGeeX** 技术增强，深度集成至 IDE 和 DevOps 流程。")

    st.subheader("🖼️ 四、多模态生成模型")
    multimodal_gen = pd.DataFrame({
        "模型系列": ["文心一格", "语音合成 TTS", "语音识别 ASR"],
        "能力": [
            "文生图、图生图、风格迁移",
            "高自然度语音生成",
            "高准确率转写"
        ],
        "状态": [
            "支持中文提示词优化，媲美 Midjourney 中文场景",
            "支持情感控制、方言（粤语、四川话等）",
            "支持会议、客服、医疗等垂直场景"
        ]
    })
    st.dataframe(multimodal_gen, use_container_width=True, hide_index=True)
    st.markdown("> 图像/语音模型可通过 **百度智能云 AI 开放平台** 调用。")

    st.subheader("🔤 五、嵌入与向量模型")
    embedding_models = pd.DataFrame({
        "模型": ["ERNIE Embedding v3", "多模态嵌入模型"],
        "输入": ["文本（最长 8K）", "图像+文本"],
        "输出维度": ["1024 / 2048", "统一向量空间"],
        "应用": ["语义搜索、聚类、RAG", "跨模态检索（如“找类似这张图的商品”）"]
    })
    st.dataframe(embedding_models, use_container_width=True, hide_index=True)

    st.subheader("🧪 六、实验性与行业定制模型")
    industry_models = pd.DataFrame({
        "模型": ["交通大模型", "法律大模型", "医疗大模型", "教育大模型"],
        "领域": ["智慧交通", "司法", "健康", "K12/高校"],
        "特点": [
            "实时路况预测、信号灯优化",
            "法条检索、文书生成、类案推送",
            "症状问答、报告解读（未大规模开放）",
            "个性化习题推荐、作文批改"
        ]
    })
    st.dataframe(industry_models, use_container_width=True, hide_index=True)

    st.subheader("🧩 七、底层技术栈（全栈自研）")
    stack = pd.DataFrame({
        "层级": ["芯片", "框架", "模型", "平台", "应用"],
        "技术": [
            "昆仑芯（Kunlun）AI 芯片（第二代已量产）",
            "飞桨 PaddlePaddle 3.0（专为大模型优化）",
            "文心大模型 4.0 / 4.0 Turbo",
            "千帆大模型平台、文心智能体平台",
            "文小言、文心一格、文心快码、百度搜索 AI"
        ]
    })
    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.subheader("🌐 访问方式")
    access_baidu = pd.DataFrame({
        "平台": ["文心智能体平台", "百度智能云千帆", "文小言（网页版）", "文心快码"],
        "地址": [
            "[https://agents.baidu.com](https://agents.baidu.com/)",
            "[https://cloud.baidu.com/product/qianfan.html](https://cloud.baidu.com/product/qianfan.html)",
            "[https://yiyan.baidu.com](https://yiyan.baidu.com/)",
            "集成至 VS Code / IDEA 插件"
        ],
        "用途": [
            "免费创建智能体（支持 3.5 / 4.0）",
            "企业级 API 调用、私有化部署",
            "直接对话体验（原“文心一言”）",
            "智能编程"
        ]
    })
    st.dataframe(access_baidu, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：百度模型战略亮点")
    st.markdown("""
    - **全栈自研**：从芯片到应用闭环，降低外部依赖。
    - **智能体优先**：将大模型能力封装为可执行 Agent，推动落地。
    - **免费开放**：文心智能体平台允许用户 **免费使用 3.5 / 4.0 构建智能体**。
    - **行业深耕**：农业、体育、金融、城市治理等场景已有标杆案例。
    """)

    st.warning("""
    📌 **注意**：
    - “文心一言”品牌已升级为 **“文小言”**，但底层仍为文心大模型。
    - **文心 5.0 尚未发布**，当前最强公开模型为 **4.0 Turbo**。
    """)
# ----------------------
# Cohere Tab
# ----------------------
with tab_cohere:
    st.header("🇨🇦 Cohere 企业级大模型生态")

    st.markdown("""
    > 截至 **2025 年 12 月**，Cohere 聚焦 **“为企业打造可部署、可定制、高安全的 LLM 解决方案”**，  
    > 与通用模型形成差异化竞争，主打 **RAG 原生优化 + 多语言 + 低成本私有化部署**。
    """)

    st.subheader("🧠 一、Command 系列主干模型（核心产品）")

    st.markdown("### ✅ 1. 旗舰推理模型：Command A Reasoning")
    st.markdown("""
    - **发布时间**：2025 年 8 月  
    - **上下文窗口**：
        - 单卡（H100/A100）：**128K tokens**
        - 双卡+：**256K tokens**
    - **核心能力**：
        - 专为 **高安全、高可控机构** 设计（如金融、政府、医疗）
        - 在 **BFCL-v3、Tau-bench、Deep Research Bench** 等企业级基准中 **超越 gpt-oss-120b、Mistral Magistral Medium**
        - 支持 **自定义 token 预算**，动态平衡精度与吞吐
        - 内置 **未成年人保护、自残防护等五大高风险内容过滤**
    - **部署方式**：
        - Cohere 平台 API
        - **私有化 / 本地部署**（支持气隙环境）
        - Hugging Face（部分版本）
    """)
    st.info("""
    💡 **典型应用**：Cohere 展示的 **Deep Research 系统** 可在数分钟内生成结构化研究报告，
    在 RACE 英文评测中得分行业第一。
    """)

    st.markdown("### ✅ 2. 高性价比企业模型：Command R+ 系列")
    command_r = pd.DataFrame({
        "模型": ["Command R7B", "Command R+", "Command R (原版)"],
        "发布时间": ["2024 年 12 月", "2024 年中", "2023 年"],
        "上下文": ["**128K tokens**", "128K tokens", "128K tokens"],
        "特点": [
            "- 最小 R 系列模型\n- 可在 **MacBook、低端 GPU、CPU** 运行\n- 原生支持 **内联引用（RAG）**\n- 多语言、数学、代码能力强",
            "- 强化 RAG 与工具调用\n- 支持 **24 种语言**（10 种核心 + 13 种快速适配）\n- 集成 **C-RAG 6.0、MV 三向量模型、RI Rank 排序**",
            "基础 RAG 优化模型"
        ],
        "适用场景": [
            "客服、HR 自动化、边缘部署",
            "企业知识库问答、跨语言客服",
            "中小企业入门"
        ]
    })
    st.dataframe(command_r, use_container_width=True, hide_index=True)

    st.markdown("""
    📊 **性能亮点**：
    - 多语言理解准确率 **89.7%**（超越 Llama 2 70B 的 85.3%）
    - 结合知识库的回答准确率达 **92.1%**（接近人类水平）
    - 编程任务准确率 **86.19%–87.52%**（适合自动化测试用例生成）
    """)

    st.markdown("### ✅ 3. 轻量化模型：Command Lite 系列")
    st.markdown("""
    - **定位**：超低成本、高并发场景（如聊天机器人、日志分析）
    - **特点**：
        - 参数规模小，单卡可承载数千 QPS
        - 支持基础多语言和 RAG
        - 价格仅为 Command R+ 的 **1/3**
    """)

    st.subheader("🖼️ 二、多模态与视觉模型")
    st.markdown("### ✅ Command A Vision")
    st.markdown("""
    - **发布时间**：2025 年 7 月
    - **能力**：
        - 从 **PDF、图表、扫描件** 等非结构化文档中提取信息
        - 支持图文联合理解与问答
    - **部署**：
        - 开放权重（供企业私有部署）
        - 目标客户：希望摆脱闭源模型（如 GPT-4V）的企业
    """)

    st.subheader("🔤 三、嵌入（Embedding）与重排序模型")
    embed_rerank = pd.DataFrame({
        "模型": ["Embed V3", "Rerank 3.5"],
        "能力": ["文本 → 向量", "提升搜索相关性"],
        "语言支持": ["英语、中文、法语等主流语言", "**100+ 语言**（含阿拉伯语、印地语、日语、韩语等）"]
    })
    st.dataframe(embed_rerank, use_container_width=True, hide_index=True)
    st.info("💡 **Rerank 3.5 是企业搜索系统的标配**，显著提升 RAG 准确率。")

    st.subheader("⚙️ 四、核心技术优势")
    core_tech = pd.DataFrame({
        "维度": ["架构优化", "企业安全", "成本效率", "生态整合"],
        "说明": [
            "原生 128K 上下文，支持处理 **约 300 页文档**（法律合同、科研论文）",
            "- 支持 **Microsoft Azure、Google Cloud、AWS** 部署\n- **私有化微调**，确保数据不出域\n- 通过 **ISO 27001、SOC 2、HIPAA** 等认证",
            "- **单张 A100 即可运行** Command R+\n- 成本仅为同类旗舰模型的 **1/3–1/4**",
            "- 兼容 **LlamaIndex、LangChain**\n- 提供 **Cohere Platform SDK**（Python/JS）"
        ]
    })
    st.dataframe(core_tech, use_container_width=True, hide_index=True)

    st.subheader("💰 五、定价与访问方式")
    pricing_cohere = pd.DataFrame({
        "模型": ["Command R7B", "Command R+", "Command A Reasoning"],
        "输入价格（每百万 tokens）": ["$0.0375", "~$0.50", "$3.00+"],
        "输出价格": ["$0.15", "~$1.50", "$15.00+"],
        "备注": [
            "最便宜，适合高频调用",
            "企业主力",
            "高端推理，按需报价"
        ]
    })
    st.dataframe(pricing_cohere, use_container_width=True, hide_index=True)

    st.markdown("""
    🌐 **访问平台**：
    - **Cohere Platform**：[https://cohere.com/platform](https://cohere.com/platform)
    - **Hugging Face**：部分模型开放（如 Command R7B）
    - **Azure Marketplace**：已上架，支持一键部署
    """)

    st.subheader("🏢 六、典型企业应用场景")
    st.markdown("""
    1. **智能客服**  
       - 对接产品文档，实时生成退换货政策解答（如电商平台）
    2. **内部知识管理**  
       - 快速检索跨部门报告中的关键数据，提升决策效率
    3. **全球化营销**  
       - 一键生成西班牙语、法语等本地化广告文案，保持品牌调性
    4. **高质量翻译**  
       - 法律合同、技术手册的精准翻译（结合 Rerank 3.5）
    5. **自动化工作流**  
       - 生成 Python/SQL 脚本、将 Excel 数据转为可视化报告
    """)

    st.subheader("💼 七、公司动态（2025 年）")
    st.markdown("""
    - **D 轮融资**：2025 年 8 月完成 **5 亿美元** 融资（Radical Ventures、Inovia Capital 领投，NVIDIA、AMD、Salesforce 跟投），估值 **55 亿美元**
    - **新平台 North**：推出基于智能体的 AI 生产力平台，实现业务系统间数据自动同步
    - **收购 Ottogrid**：强化市场研究智能体能力
    - **加拿大政府投资**：获 **2.4 亿美元** 支持，建设本土 AI 数据中心
    """)

    st.subheader("✅ 总结：Cohere 模型战略定位")
    strategy = pd.DataFrame({
        "维度": ["目标客户", "技术差异", "生态位"],
        "策略": [
            "中小企业 → 大型企业 → 政府/金融/医疗等高监管行业",
            "**RAG 原生优化 + 企业安全 + 多语言 + 低成本部署**",
            "填补 **开源模型（Llama）** 与 **闭源旗舰（GPT-4、Claude Opus）** 之间的 **高性价比企业级空白**"
        ]
    })
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.success("""
    📌 **推荐选择**：
    - **成本敏感型应用** → **Command R7B**
    - **复杂知识问答** → **Command R+**
    - **高安全推理任务** → **Command A Reasoning**
    """)

# ----------------------
# Google DeepMind Tab
# ----------------------
with tab_deepmind:
    st.header("🔬 Google DeepMind 模型体系概览")

    st.markdown("""
    > 截至 **2025 年 12 月**，DeepMind 已从早期以 **强化学习与科学发现** 为主的实验室，全面转型为 **通用人工智能（AGI）、具身智能、世界模型、生物计算、机器人控制** 等多领域的全球领导者。
    """)

    st.subheader("🧠 一、通用大语言模型与推理系统")

    st.markdown("### ✅ Gemini 系列（DeepMind 主导开发）")
    gemini_models = pd.DataFrame({
        "模型": ["Gemini 2.5 Pro “I/O”", "Gemini 2.0 Ultra", "Gemini Nano"],
        "发布时间": ["2025 年 5 月", "2025 年初", "2023–2025"],
        "类型": ["MoE（激活 ~48B）", "稠密 + MoE 混合", "轻量稠密"],
        "多模态": ["✅（文本+图像+音频+视频）", "✅", "✅（部分）"],
        "核心能力": [
            "- 实时工具调用\n- 长上下文（1M+ tokens）\n- 代码生成 SOTA",
            "全球最强多模态模型之一，支持复杂任务规划",
            "运行于 Pixel 手机，支持离线 AI 功能"
        ]
    })
    st.dataframe(gemini_models, use_container_width=True, hide_index=True)
    st.info("💡 Gemini 已成为 **Android 16、Google Assistant、Workspace** 的默认 AI 引擎。")

    st.subheader("🤖 二、具身智能与机器人模型（2025 年重点方向）")

    st.markdown("### ✅ Gemini Robotics 系列")
    robotics_models = pd.DataFrame({
        "模型": ["Gemini Robotics 1.5", "Gemini Robotics-ER 1.5", "Gemini Robotics On-Device"],
        "发布时间": ["2025 年 9 月", "2025 年 9 月", "2025 年 6 月"],
        "架构": ["视觉-语言-动作（VLA） + 推理链", "具象推理（Embodied Reasoning）", "本地化 VLA 模型"],
        "能力亮点": [
            "- 生成自然语言解释自身决策\n- 失败率从 44.5% → 22%\n- 支持 **零样本跨机器人迁移**（ALOHA → Apollo）",
            "- 在 15 项基准测试中平均 **62.8 分**（第二名 60.6）\n- 可完成“根据天气打包行李”等多步任务",
            "- 无需云端，直接控制实体机器人\n- 支持 NVIDIA GR00T N1 人形机器人"
        ]
    })
    st.dataframe(robotics_models, use_container_width=True, hide_index=True)
    st.success("🤝 **合作平台**：\n- 与 **NVIDIA、Disney** 联合开发开源物理引擎 **Newton**（2025 年 3 月）；\n- 支持 **GR00T N1**（全球首款开源人形机器人功能模型）。")

    st.subheader("🌍 三、世界模型（World Models）——虚拟环境生成")

    st.markdown("### ✅ Genie 系列")
    genie_models = pd.DataFrame({
        "模型": ["Genie 3", "Genie 2", "Genie 1"],
        "发布时间": ["2025 年 8 月", "2024 年 12 月", "2024 年 2 月"],
        "输入": ["文本提示", "单张图片 + 文字", "文本"],
        "输出": ["**720p、24fps、分钟级一致性的 3D 世界**", "无限种类可玩 3D 世界", "可玩 2D/3D 游戏"],
        "性能": [
            "- 可实时导航\n- 支持用户返回历史位置（记忆一致性）",
            "升级版 Genie 1，支持更复杂物理交互",
            "首次实现“一句话生成游戏”"
        ]
    })
    st.dataframe(genie_models, use_container_width=True, hide_index=True)
    st.warning("🔒 **访问方式**：仅向 **科研机构与创意工作者** 提供研究预览（非公开 API）。")

    st.subheader("🧬 四、生物与科学计算模型（DeepMind 传统强项）")

    st.markdown("### ✅ AlphaFold 系列")
    alphafold_models = pd.DataFrame({
        "模型": ["AlphaFold 3", "AlphaFold DB"],
        "发布时间": ["2024 年 5 月", "持续更新"],
        "能力": ["预测 **蛋白质、DNA、RNA、小分子、离子** 的结构与相互作用", "包含 **超 2 亿种蛋白质结构**"],
        "开放状态": ["✅ **开源**（2024 年 11 月发布权重 + Web Server）", "✅ 免费查询（EBI 合作）"]
    })
    st.dataframe(alphafold_models, use_container_width=True, hide_index=True)
    st.info("💊 **临床进展**：2025 年 1 月，Demis Hassabis 宣布 **AI 设计药物进入临床试验阶段**（通过子公司 Isomorphic Labs）。")

    st.subheader("其他科学模型")
    other_science_models = pd.DataFrame({
        "模型": ["AlphaCode 2", "AlphaTensor", "FermiNet / PauliNet", "Control for Fusion"],
        "领域": ["编程竞赛", "数学算法", "量子化学", "核聚变"],
        "成就": [
            "在 Codeforces 达 **Top 15%** 人类水平",
            "发现更快的矩阵乘法算法",
            "精确求解薛定谔方程",
            "2022 年实现 AI 控制等离子体稳定"
        ]
    })
    st.dataframe(other_science_models, use_container_width=True, hide_index=True)

    st.subheader("🗣️ 五、对话与代理模型")
    dialogue_agents = pd.DataFrame({
        "模型": ["Sparrow", "Pi", "SIMI"],
        "发布时间": ["2022 年 9 月", "2023 年 5 月", "2024 年 3 月"],
        "特点": [
            "安全对齐聊天机器人，强调事实性与有害性控制",
            "“情感陪伴型” AI，定位为“朋友”而非工具",
            "可在多个虚拟世界中执行指令的通用智能体"
        ]
    })
    st.dataframe(dialogue_agents, use_container_width=True, hide_index=True)

    st.subheader("🧩 六、架构与基础设施创新")
    infra_innovations = pd.DataFrame({
        "技术": ["Zipper", "V2A", "SignGemma", "Centaur"],
        "说明": [
            "模块化架构，由多个单模态预训练解码器组成，支持灵活组合",
            "根据画面或提示词为视频配音，支持情感与风格控制",
            "最强手语翻译模型，将手语转化为口语文本，**开源**并加入 Gemma 家族",
            "与 Helmholtz AI、普林斯顿合作，**首次预测人类行为**的大模型"
        ]
    })
    st.dataframe(infra_innovations, use_container_width=True, hide_index=True)

    st.subheader("🏢 七、组织与生态")
    st.markdown("""
    - **Isomorphic Labs**：DeepMind 分拆的生物医药公司，专注 AI 药物研发；
    - **收编 Windsurf 团队**：2025 年 7 月，吸纳原 OpenAI 收购失败的 AI 初创核心成员；
    - **AGI 安全框架**：2025 年 4 月发布全球 AGI 安全倡议，呼吁建立跨国监管机制。
    """)

    st.subheader("🔒 总结：Google DeepMind 模型战略")
    summary = pd.DataFrame({
        "领域": ["通用 AI", "具身智能", "虚拟世界", "生命科学", "编程/数学", "无障碍技术"],
        "代表模型": ["Gemini 2.5 Pro", "Gemini Robotics 1.5", "Genie 3", "AlphaFold 3", "AlphaCode 2, AlphaTensor", "SignGemma"],
        "开源状态": ["❌", "❌", "❌（研究预览）", "✅", "❌", "✅"],
        "应用方向": ["Google 产品全家桶", "机器人、自动驾驶", "元宇宙、智能体训练", "药物研发、基础科研", "竞赛、算法优化", "手语翻译、包容性 AI"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.success("""
    📌 **关键趋势**：
    - **从“单一任务突破”转向“通用智能系统”**；
    - **闭源为主，但关键科学成果（如 AlphaFold）选择开源以推动领域进步**；
    - **深度绑定 Google Cloud 与硬件（TPU v5e/v6）**，形成技术闭环。
    """)
# ----------------------
# Google AI 模型全家桶 Tab
# ----------------------
with tab_google_ai:
    st.header("🟣 Google AI 模型全家桶（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，Google 已构建以 **Gemini** 为核心的完整生成式 AI 产品生态，  
    > 覆盖大语言模型、多模态、图像、视频、嵌入、开源轻量、医疗等场景。  
    > **PaLM 系列已全面退役**，战略重心完全转向 Gemini。
    """)

    st.subheader("🧠 一、核心大语言与多模态模型（Gemini 系列）")
    st.markdown("> 原生支持文本、图像、音频、视频、代码五模态")

    gemini_main = pd.DataFrame({
        "模型名称": [
            "Gemini 3 Pro",
            "Gemini 2.5 Pro",
            "Gemini 2.5 Flash",
            "Gemini 2.5 Flash Image",
            "Gemini 2.5 Flash-Lite",
            "Gemini 2.0 Flash / Flash-Lite",
            "Gemini 1.5 Pro",
            "Gemini 1.5 Flash / Flash-8B"
        ],
        "类型": [
            "多模态推理旗舰",
            "高性能多模态",
            "快速通用",
            "图像生成+编辑",
            "轻量高吞吐",
            "基础多模态",
            "超长上下文",
            "快速轻量"
        ],
        "上下文": [
            "1M tokens",
            "1M tokens",
            "1M tokens",
            "-",
            "-",
            "-",
            "**2M tokens**",
            "1M / 1M tokens"
        ],
        "特点": [
            "自适应思考、代理工作流优化、集成 grounding，LMArena 排行第一（1501 Elo）",
            "复杂推理、编码、自适应思考",
            "低延迟、可控思考预算",
            "对话式编辑、角色一致性、多图融合",
            "大规模部署优化",
            "高性价比，适合通用任务",
            "支持 200 万 token 输入（历史最强上下文）",
            "Flash-8B 为 80 亿参数低成本版"
        ]
    })
    st.dataframe(gemini_main, use_container_width=True, hide_index=True)
    st.info("💡 注：**Gemini 1.0 Ultra** 曾在早期宣传中出现，但未作为独立模型在 API 中长期提供，已被后续版本取代。")

    st.subheader("🌱 二、开源轻量模型（Gemma 系列）")
    st.markdown("> 开源、可本地部署，适合边缘设备和研究")

    gemma_models = pd.DataFrame({
        "模型": [
            "Gemma 3n",
            "Gemma 3",
            "Gemma 2",
            "Gemma (1)",
            "CodeGemma",
            "PaliGemma",
            "ShieldGemma 2",
            "MedGemma",
            "TxGemma / T5Gemma / MedSigLIP"
        ],
        "参数规模": [
            "超小",
            "-",
            "-",
            "2B / 7B",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        "模态": [
            "文本+图像+音频+视频",
            "文本+图像",
            "文本",
            "文本",
            "代码",
            "视觉+语言",
            "文本+图像",
            "医学文本+图像",
            "专用"
        ],
        "特点": [
            "支持 140+ 语言，低资源设备优化",
            "128K 上下文，多语言",
            "生成/摘要/抽取",
            "基础开源 LLM",
            "代码补全、生成、理解",
            "SigLIP + Gemma 融合",
            "内容安全审核",
            "Gemma 3 的医学微调版",
            "医疗预测、编码器-解码器、医学图文嵌入"
        ]
    })
    st.dataframe(gemma_models, use_container_width=True, hide_index=True)

    st.subheader("🖼️ 三、图像生成模型（Imagen 系列）")
    imagen_models = pd.DataFrame({
        "模型": [
            "Imagen 4 for Generation / Fast / Ultra",
            "Imagen 3 (001, 002, Fast, Editing)",
            "虚拟试穿 / Product Recontext"
        ],
        "状态": ["正式", "正式", "预览"],
        "特性": [
            "高质量、低延迟、强提示遵循",
            "基础生成与编辑",
            "电商场景专用（Vertex AI）"
        ]
    })
    st.dataframe(imagen_models, use_container_width=True, hide_index=True)

    st.subheader("🎥 四、视频生成模型（Veo 系列）")
    veo_models = pd.DataFrame({
        "模型": ["Veo 3.1 Generate / Fast", "Veo 3 / Veo 2", "Veo 实验版"],
        "状态": ["正式", "正式/预览", "实验"],
        "能力": [
            "高质量文本/图像 → 视频",
            "支持修复、扩绘、多轮生成",
            "测试新功能"
        ]
    })
    st.dataframe(veo_models, use_container_width=True, hide_index=True)

    st.subheader("🔤 五、嵌入（Embedding）模型")
    embedding_google = pd.DataFrame({
        "模型": [
            "text-embedding-004",
            "gemini-embedding-exp",
            "Multimodal Embeddings",
            "embedding-gecko-001"
        ],
        "输入": ["文本（8K tokens）", "文本", "图像", "文本（1K tokens）"],
        "输出维度": ["-", "-", "-", "-"],
        "用途": [
            "语义搜索、聚类",
            "实验性长上下文嵌入",
            "图像检索、分类",
            "旧版嵌入"
        ]
    })
    st.dataframe(embedding_google, use_container_width=True, hide_index=True)

    st.subheader("🏥 六、医疗专用模型")
    med_models = pd.DataFrame({
        "模型": ["MedLM-medium / large-large", "MedGemma / MedSigLIP"],
        "状态": ["❌ **已弃用**（2025-09-29 停用）", "✅ 活跃"],
        "说明": [
            "HIPAA 合规医疗问答",
            "新一代医学多模态模型"
        ]
    })
    st.dataframe(med_models, use_container_width=True, hide_index=True)

    st.subheader("📚 七、历史模型（已逐步淘汰）")
    legacy_models = pd.DataFrame({
        "模型": ["PaLM 2", "PaLM / PaLM-E"],
        "说明": [
            "Gemini 前代主力模型，曾用于 Bard，现被 Gemini 全面取代",
            "早期大模型及具身智能实验版，不再主推"
        ]
    })
    st.dataframe(legacy_models, use_container_width=True, hide_index=True)
    st.warning("⚠️ **注意**：Google 已将战略重心全面转向 **Gemini 生态**，PaLM 系列不再更新。")

    st.subheader("🌐 访问方式")
    access_google = pd.DataFrame({
        "平台": ["普通用户", "开发者测试", "企业部署"],
        "地址/方式": [
            "[https://gemini.google.com](https://gemini.google.com/)",
            "[Google AI Studio](https://aistudio.google.com/)",
            "[Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)"
        ]
    })
    st.dataframe(access_google, use_container_width=True, hide_index=True)

    st.success("""
    ✅ **总结**：  
    Google 当前模型体系以 **Gemini** 为核心，辅以 **Gemma（开源）**、**Imagen（图像）**、**Veo（视频）**、**Embedding** 等垂直模型，  
    形成完整的生成式 AI 生态。**PaLM 2 已退出主流服务**，仅作为历史技术存在。
    """)

# ========================
# 🇨🇳 华为盘古（你提供的内容）
# ========================
with tab_huawei:
    st.header("🇨🇳 华为盘古大模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，华为以 **“AI for Industries”** 为核心战略，  
    > 构建覆盖 **自然语言、计算机视觉、多模态、科学计算、时序预测** 的全栈 AI 体系，  
    > 深度聚焦 **矿山、电力、气象、金融、制造等 30+ 行业**。  
    > 采取 **“闭源主干 + 开源轻量”** 策略，全面支持 **昇腾（Ascend）国产算力**。
    """)

    st.subheader("🧠 一、盘古大模型主干系列（闭源，企业级）")

    pangu_main = pd.DataFrame({
        "模型版本": ["盘古大模型 5.5", "盘古大模型 5.0", "盘古大模型 3.0"],
        "发布时间": ["2025 年 6 月 20 日", "2024 年 6 月 21 日", "2023 年 7 月"],
        "核心升级": [
            "- NLP、CV、多模态、预测、科学计算五大基础模型全面升级\n- 支持 **智能驾驶世界模型**（生成行车视频+激光雷达数据）\n- Agent 工具调用能力增强，幻觉率显著降低",
            "- 参数规模覆盖十亿至万亿级\n- 首次与 **HarmonyOS NEXT** 深度集成\n- 推出 **小艺输入法 Beta**（Mate 60 系列专属）",
            "- 首提 **L0（基础）+ L1（行业）+ L2（场景）三层架构**\n- 首个商用：**山东能源矿山大模型**（覆盖采煤、安监等 21 场景）"
        ],
        "行业应用": [
            "自动驾驶、智能制造、金融风控、气象预报",
            "车载助手、智慧政务、钢铁/能源行业",
            "矿山、电力、金融、医药、铁路"
        ]
    })
    st.dataframe(pangu_main, use_container_width=True, hide_index=True)
    st.info("💡 **盘古 5.5 是当前最强版本**，在 **SuperCLUE 2025 年 5 月榜单中并列中国第一**。")

    st.subheader("🔓 二、开源模型（GitHub / GitCode）")
    st.markdown("> 华为于 **2025 年 7 月起** 开源多个轻量级盘古模型，全部基于 **昇腾 NPU** 优化")

    open_models = pd.DataFrame({
        "模型名称": ["Pangu-7B", "Pangu-Pro-MoE", "openPangu-Ultra-MoE-718B-V1.1"],
        "参数": ["7B", "总 72B / 激活 ~12B", "总 718B / 激活 39B"],
        "架构": ["稠密", "混合专家（MoE）", "MoE"],
        "开源地址": [
            "[GitCode](https://gitcode.com/)",
            "[GitCode](https://gitcode.com/)",
            "[GitCode](https://gitcode.com/openpangu)"
        ],
        "特点": [
            "- 4 张消费级显卡即可部署\n- 支持快速思考/慢速思考统一推理",
            "高性能行业微调基座",
            "- 全球最大开源中文 MoE 模型之一\n- Agent 工具调用 SOTA，幻觉率 <2%"
        ]
    })
    st.dataframe(open_models, use_container_width=True, hide_index=True)
    st.success("✅ 所有开源模型均提供 **完整权重、训练代码、昇腾推理加速库**，支持 **ModelArts Studio 一键部署**。")

    st.subheader("🏭 三、行业专用模型（已落地 30+ 行业，500+ 场景）")
    industry_cases = pd.DataFrame({
        "行业": ["矿山", "气象", "金融", "制造", "政务", "车载"],
        "模型/解决方案": [
            "盘古矿山大模型",
            "盘古气象大模型",
            "盘古金融大模型 + 深交所法规大模型",
            "盘古钢铁大模型（湖南钢铁）",
            "盘古政务大模型（深圳福田）",
            "小艺智慧助手（HarmonyOS）"
        ],
        "成果": [
            "覆盖 9 大专业 21 场景，提升安全效率 30%",
            "台风轨迹预测准确率 **90%+**，马达加斯加上线每日 10 日预报",
            "问答准确率 **>90%**，服务数十万网点柜员",
            "全球首发，优化配煤、能耗",
            "“小福”助手提升办事效率 50%",
            "听懂自然语言，化身“懂车艺”"
        ]
    })
    st.dataframe(industry_cases, use_container_width=True, hide_index=True)

    st.subheader("🧪 四、前沿架构与研究模型")
    research_models = pd.DataFrame({
        "模型/技术": ["盘古-π（Pangu-π）", "云山大模型", "盘古科学计算大模型"],
        "说明": [
            "- 改进 Transformer，增加非线性\n- 降低特征塌陷，提升表达能力\n- 在 7B 规模超越 LLaMA，推理加速 **10%**",
            "基于盘古-π 的 **金融法律专用模型**，在多个基准测试中领先",
            "用于流体力学、材料模拟、气候建模等 HPC 场景"
        ]
    })
    st.dataframe(research_models, use_container_width=True, hide_index=True)
    st.caption("🔬 由 **华为诺亚方舟实验室** 与 **2012 实验室** 联合研发，田奇任首席科学家。")

    st.subheader("⚙️ 五、技术底座与平台")
    tech_stack = pd.DataFrame({
        "组件": ["训练芯片", "开发平台", "推理加速", "生态合作"],
        "说明": [
            "昇腾 910B × 2000+（千亿模型训练 >2 个月）",
            "**ModelArts Studio**（唯一企业入口）\n- 支持数据工程、模型训练、Agent 开发、RAG 构建",
            "MindSpore + CANN（昇腾软件栈），端边云协同",
            "与循环智能、鹏城实验室、高校联合开发"
        ]
    })
    st.dataframe(tech_stack, use_container_width=True, hide_index=True)

    st.subheader("🌐 访问方式")
    access_info = pd.DataFrame({
        "类型": ["企业服务", "开源模型", "文档与案例"],
        "平台": [
            "华为云 ModelArts Studio",
            "GitCode（国内） / GitHub（镜像）",
            "华为云盘古官网"
        ],
        "地址": [
            "[https://www.huaweicloud.com/product/modelarts.html](https://www.huaweicloud.com/product/modelarts.html)",
            "[https://gitcode.com/openpangu](https://gitcode.com/openpangu)",
            "[https://www.huaweicloud.com/pangu](https://www.huaweicloud.com/pangu)"
        ]
    })
    st.dataframe(access_info, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：华为盘古大模型战略特点")
    strategy_summary = pd.DataFrame({
        "维度": ["定位", "开源", "闭源", "全栈自研", "落地能力"],
        "策略": [
            "**AI for Industries** —— 不做通用聊天，专注产业价值",
            "轻量模型（7B–718B）全面开源，支持国产算力（昇腾）",
            "5.0/5.5 等旗舰模型仅限企业 API 调用",
            "芯片（昇腾）→ 框架（MindSpore）→ 模型（盘古）→ 应用（行业方案）",
            "**30+ 行业、500+ 场景**，全球首个矿山大模型商用"
        ]
    })
    st.dataframe(strategy_summary, use_container_width=True, hide_index=True)

    st.success("""
    📌 **推荐路径**：
    - **开发者/研究者** → 使用 **Pangu-7B / openPangu-718B** 开源模型；
    - **企业客户** → 通过 **ModelArts Studio** 调用 **盘古 5.5** 行业 API。
    """)

# ========================
# 🦛 Meta AI（新增）
# ========================
with tab_meta:
    st.header("🦛 Meta AI 模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，Meta 采取 **“开源权重 + 闭源服务”双轨战略**：
    > 
    > - **Llama 系列**以开放权重（Open Weights）形式发布，允许研究和商用（需申请许可），但**不开放训练数据、完整训练代码或推理优化细节**。
    > - 同时，Meta 正秘密开发下一代 **闭源旗舰模型 Avocado**，计划于 **2026 年 Q1 发布**，标志着其从“开源馈赠”向“商业化变现”的重大转向。
    """)

    st.subheader("🦙 一、Llama 系列（开放权重模型）")
    st.markdown("> ✅ **开放权重 ≠ 完全开源**：可下载模型权重，但训练数据、tokenizer 细节、RLHF 流程、高效推理内核仍为闭源。")

    llama_models = pd.DataFrame({
        "模型名称": ["Llama 4", "Llama 3.1", "Llama 3", "Llama 2"],
        "参数规模": [
            "MoE（约 128 专家，激活 8）",
            "8B / 70B / **405B**",
            "8B / 70B",
            "7B / 13B / 70B"
        ],
        "上下文窗口": ["128K", "128K", "8K → 128K（微调版）", "4K"],
        "多模态": ["❌", "❌", "❌", "❌"],
        "架构特点": [
            "全系首次采用 Mixture of Experts（MoE）",
            "稠密模型，性能超越 GPT-4o（在部分基准）",
            "首次支持 128K 上下文（via RoPE 扩展）",
            "首个商用许可开放模型"
        ],
        "发布时间": ["2025 年 4 月", "2025 年 3 月", "2024 年 4 月", "2023 年 7 月"]
    })
    st.dataframe(llama_models, use_container_width=True, hide_index=True)
    st.warning("💡 **Llama 4 表现未达预期**：内部承认“性能与热度均落后竞品”，成为 Meta 转向闭源 Avocado 的催化剂。")

    st.subheader("📸 二、多模态与视觉模型")
    vision_models = pd.DataFrame({
        "模型名称": ["SAM 3", "SAM 2 / SAM 1", "DINOv2", "ImageBind"],
        "类型": ["图像分割", "视频/图像分割", "自监督视觉编码器", "多模态对齐"],
        "输入/输出": [
            "图像 → 分割掩码",
            "支持视频时序一致性",
            "图像 → 嵌入",
            "文本/图像/音频/深度等 → 统一嵌入空间"
        ],
        "特点": [
            "**零样本 + 自然语言指令分割**（如“分割所有狗”）",
            "开源，广泛用于 CV 社区",
            "无需标注预训练，适用于下游任务",
            "支持跨模态检索（如用音频搜图）"
        ]
    })
    st.dataframe(vision_models, use_container_width=True, hide_index=True)
    st.caption("🔒 这些模型 **权重开源**，但 **训练基础设施和大规模数据集（如 DataComp）不公开**。")

    st.subheader("🧠 三、下一代闭源旗舰：Avocado（开发中）")
    st.markdown("> ⚠️ **非 Llama 系列**，定位为 **Llama 的“精神继任者”**，但**完全闭源**。")

    avocado_info = pd.DataFrame({
        "项目": [
            "代号",
            "发布时间",
            "定位",
            "架构",
            "能力重点",
            "分发方式",
            "商业化",
            "资源投入"
        ],
        "详情": [
            "Avocado（牛油果）",
            "**2026 年 Q1**（原定 2025 年底，因性能未达标推迟）",
            "Frontier-level 闭源大模型，对标 GPT-5、Gemini 3 Ultra",
            "Llama 改进 + 全新多模态编码器",
            "长上下文、工具调用、推理速度、智能体协作",
            "**仅通过 API 和托管服务**（不再开放权重）",
            "通过 API 调用、企业托管、Meta AI 广告增强变现",
            "- 150 亿美元人才收购（含 Scale AI CEO Alexandr Wang 任首席 AI 官）\n- 27 亿美元建 Hyperion 数据中心（路易斯安那州）"
        ]
    })
    st.dataframe(avocado_info, use_container_width=True, hide_index=True)
    st.info("💬 Meta 发言人称：“模型训练按计划进行，无重大时间表变更”——但多方信源证实已延期。")

    st.subheader("🛠️ 四、专用模型与工具")
    tools = pd.DataFrame({
        "模型/工具": ["Code Llama", "Llama Guard", "Prompt Guard", "Chameleon", "Vibes"],
        "用途": [
            "代码生成（Python, C++, etc.）",
            "内容安全过滤（输入/输出审核）",
            "检测越狱提示（jailbreak prompts）",
            "早期多模态实验模型（文本+图像联合生成）",
            "AI 生成短视频平台（类似 Sora）"
        ],
        "开源状态": ["✅ 权重开源（基于 Llama 2/3）", "✅ 开源，支持自定义策略", "✅ 开源", "❌ 未公开权重", "❌ 闭源产品，市场反响平淡"]
    })
    st.dataframe(tools, use_container_width=True, hide_index=True)

    st.subheader("🌐 五、访问与部署方式")
    access_meta = pd.DataFrame({
        "方式": [
            "Hugging Face / GitHub",
            "Meta AI 官方 API（2025 年推出）",
            "DeepInfra / Together.ai / Groq",
            "llama.cpp / vLLM / TensorRT-LLM"
        ],
        "说明": [
            "下载 Llama、SAM、Code Llama 等权重",
            "提供 Llama 3.1 / Llama 4 的官方 API（此前仅第三方提供）",
            "第三方云平台提供 Llama 模型推理（价格更低）",
            "社区优化推理框架（Meta 不官方维护）"
        ]
    })
    st.dataframe(access_meta, use_container_width=True, hide_index=True)
    st.warning("""
    ⚠️ **注意**：
    - 使用 Llama 商业需 **申请 Meta 许可**（[llama.meta.com](https://llama.meta.com/)）。
    - **Llama 4 MoE 模型因复杂性，部署成本极高**，社区支持有限。
    """)

    st.subheader("✅ 总结：Meta 模型战略演进")
    strategy_evolution = pd.DataFrame({
        "阶段": ["2023–2024", "2025", "2026 起"],
        "战略": ["“开源建立生态”", "“开源遇挫，转向混合”", "“全面商业化”"],
        "代表模型": ["Llama 2 / Llama 3", "Llama 4（表现平庸） + 官方 API 推出", "**Avocado**"],
        "开放性": ["✅ 开放权重", "⚠️ 权重开放，服务闭源", "❌ 完全闭源，API Only"]
    })
    st.dataframe(strategy_evolution, use_container_width=True, hide_index=True)

    st.success("""
    > Meta 正从 **AI 社区贡献者** 转变为 **封闭模型竞争者**，意图在 2026 年与 OpenAI、Google 形成“三强对决”。
    """)
# ========================
# 🔷 Microsoft AI（新增）
# ========================
with tab_microsoft:
    st.header("🔷 Microsoft AI 模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，Microsoft 采取 **“自研 + 战略合作”双轨模式**：
    > 
    > - 与 **OpenAI 深度绑定**，独家提供 GPT 系列（如 GPT-4、GPT-5）在企业级云服务中的部署（通过 **Azure OpenAI Service**）。
    > - 同时大力投入 **自研模型**（Phi、Orca、Foundry），并整合 **Anthropic、Mistral、Meta** 等第三方模型。
    > 
    > 核心优势：**“模型无关”的企业 AI 平台**——统一管理、安全部署、灵活路由。
    """)

    st.subheader("🧠 一、Azure OpenAI Service 提供的 OpenAI 模型（主力）")
    st.markdown("> 所有模型均可在 [Azure OpenAI Studio](https://oai.azure.com/) 部署，支持企业级安全、合规、私有网络。")

    azure_openai_models = pd.DataFrame({
        "模型名称": [
            "GPT-5.1",
            "GPT-5 / GPT-5-mini / GPT-5-nano",
            "GPT-4o / GPT-4o-mini",
            "GPT-4 Turbo (gpt-4-turbo)",
            "GPT-3.5-Turbo",
            "GPT-5-codex",
            "o4-mini, o3-pro, codex-mini",
            "GPT-image-1 / GPT-image-1-mini",
            "Sora"
        ],
        "类型": [
            "推理旗舰",
            "系列模型",
            "多模态全能",
            "文本主力",
            "轻量文本",
            "代码专用",
            "推理/代码子模型",
            "图像生成",
            "视频生成"
        ],
        "上下文": [
            "400K tokens",
            "128K–256K",
            "128K",
            "128K",
            "16K",
            "128K",
            "-",
            "-",
            "-"
        ],
        "多模态": ["✅", "✅", "✅", "❌", "❌", "❌", "-", "✅", "✅"],
        "发布/可用时间": [
            "2025 年 11 月",
            "2025 年 8 月起",
            "2024–2025",
            "2023–2024",
            "持续可用",
            "2025 年 9 月",
            "2025 年中",
            "2025 年 4–7 月",
            "2025 年 5 月起（预览）"
        ],
        "特点": [
            "最强编码与智能体任务，可配置推理强度",
            "覆盖高性能到轻量场景",
            "支持实时语音对话、ASR、TTS",
            "高性价比通用模型",
            "低成本基础模型",
            "专为 VS Code、CLI 优化的代码生成",
            "用于特定任务微调",
            "DALL·E 升级版，支持图像编辑、保真度控制",
            "支持图像→视频、视频→视频生成"
        ]
    })
    st.dataframe(azure_openai_models, use_container_width=True, hide_index=True)
    st.info("""
    💡 **注意**：
    - 所有 GPT 模型在 Azure 中均支持 **内容过滤、PII 检测、私有 VNET、审计日志**。
    - **GPT-5.1 是当前 Azure OpenAI 最强公开模型**（超越 GPT-4o 在编码和代理任务上）。
    """)

    st.subheader("🔬 二、Microsoft 自研模型")

    st.markdown("### ✅ 1. Phi 系列（小型高效，面向边缘/研究）")
    phi_models = pd.DataFrame({
        "模型": ["Phi-4", "Phi-3.5-mini / vision", "Phi-3-small / medium"],
        "参数": ["~10B", "3.8B / 4.2B", "7B / 14B"],
        "特点": [
            "多模态（文本+图像），支持 128K 上下文，性能接近 Llama-3-70B",
            "轻量多模态，可在手机运行",
            "平衡性能与效率，支持函数调用"
        ]
    })
    st.dataframe(phi_models, use_container_width=True, hide_index=True)
    st.caption("[开源地址：Hugging Face - microsoft/Phi](https://huggingface.co/microsoft)")

    st.markdown("### ✅ 2. Orca 系列（推理增强）")
    orca_models = pd.DataFrame({
        "模型": ["Orca-2", "Orca-Math"],
        "特点": [
            "通过模仿 GPT-4 行为提升小模型推理能力",
            "专精数学推理（AMC、MATH 数据集 SOTA）"
        ]
    })
    st.dataframe(orca_models, use_container_width=True, hide_index=True)

    st.markdown("### ✅ 3. Microsoft Foundry 模型（企业级推理平台）")
    st.markdown("""
    - 并非单一模型，而是一个 **模型调度与路由系统**：
      - 自动选择最佳模型（GPT-5、Claude、Mistral 等）响应用户请求。
      - 支持 **模型快照、A/B 测试、成本优化路由**。
      - 集成 **Realtime API（SIP/WebRTC）**，用于电话客服等场景。
    """)

    st.subheader("🤝 三、Azure AI Foundry 集成的第三方模型")
    third_party = pd.DataFrame({
        "模型提供商": ["Anthropic", "Mistral AI", "Meta", "Google"],
        "可用模型示例": [
            "Claude 3.5 Sonnet, Claude 3 Opus",
            "Mistral Large 2, Codestral",
            "Llama 3.1 405B, Llama 3 70B/8B",
            "Gemini Pro（有限集成）"
        ],
        "访问方式": [
            "Azure AI Studio → Model Catalog",
            "直接部署或通过 API",
            "需申请许可，支持 BYOC（自带容器）",
            "实验性支持"
        ]
    })
    st.dataframe(third_party, use_container_width=True, hide_index=True)
    st.success("✅ 优势：统一 API、监控、计费、安全策略，无需切换平台。")

    st.subheader("🎙️ 四、语音与多模态专用模型")
    speech_models = pd.DataFrame({
        "模型": ["GPT-realtime", "gpt-4o-transcribe-diarize", "Marin / Cedar"],
        "功能": [
            "低延迟语音对话（ASR + TTS + LLM）",
            "语音转文本 + 说话人分离",
            "新一代合成语音"
        ],
        "状态": ["GA（正式发布）", "支持 100+ 语言", "更自然、清晰"]
    })
    st.dataframe(speech_models, use_container_width=True, hide_index=True)

    st.subheader("📊 五、Copilot 生态集成模型")
    copilot_integration = pd.DataFrame({
        "产品": [
            "Microsoft Copilot（Windows）",
            "GitHub Copilot",
            "Microsoft 365 Copilot",
            "Azure AI Search"
        ],
        "使用的模型": [
            "GPT-4o、GPT-5、Claude（2025 年新增）",
            "GPT-5-codex、CodeGemma",
            "GPT-5、GPT-4o、自研 Orca",
            "GPT-4 Turbo + 嵌入模型"
        ]
    })
    st.dataframe(copilot_integration, use_container_width=True, hide_index=True)
    st.info("💡 **重大更新（2025 年 12 月）**：Copilot **不再只依赖 OpenAI**，已接入 **Anthropic Claude** 和 **Mistral** 模型，实现多模型混合推理。")

    st.subheader("🌐 访问方式汇总")
    access_ms = pd.DataFrame({
        "平台": [
            "Azure OpenAI Studio",
            "Azure AI Studio",
            "Model Catalog",
            "GitHub Models"
        ],
        "地址": [
            "[https://oai.azure.com](https://oai.azure.com/)",
            "[https://ai.azure.com](https://ai.azure.com/)",
            "Azure Portal → AI + Machine Learning",
            "[https://github.com/microsoft/Phi](https://github.com/microsoft/Phi)"
        ],
        "用途": [
            "部署 GPT 系列",
            "访问 Phi、Claude、Mistral、Llama 等",
            "浏览所有可用模型",
            "下载开源 Phi 模型"
        ]
    })
    st.dataframe(access_ms, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：Microsoft 模型战略")
    ms_strategy = pd.DataFrame({
        "类型": ["合作模型（主力）", "自研模型", "集成第三方"],
        "代表模型": ["GPT-5.1, GPT-4o, Sora", "Phi-4, Orca, Foundry", "Claude, Mistral, Llama"],
        "定位": [
            "企业级生成式 AI 核心",
            "边缘计算、推理优化、成本控制",
            "避免供应商锁定，提升灵活性"
        ]
    })
    st.dataframe(ms_strategy, use_container_width=True, hide_index=True)

    st.success("""
    > Microsoft 的核心优势在于 **“模型无关”的企业 AI 平台**——无论模型来自 OpenAI、Anthropic 还是自研，  
    > 均可在 Azure 统一管理、安全部署、智能路由。
    """)
# ========================
# 🎨 Stability AI（新增）
# ========================
with tab_stability:
    st.header("🎨 Stability AI 模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，Stability AI 已从单一图像模型公司扩展为覆盖 **图像、视频、音频、3D、文本** 的多模态开源模型提供商。
    > 
    > **核心战略**：
    > - **开源基础模型权重**（如 Stable Diffusion XL、Stable Audio 2.5），推动社区生态；
    > - **通过 API 和 SaaS 服务提供增强版/闭源功能**（如高清修复、视频生成、音频修补）；
    > - **与企业合作定制专属模型**（如 WPP 品牌音效、广告视觉生成）。
    """)

    st.subheader("🖼️ 一、图像生成模型（核心领域）")

    st.markdown("### ✅ 1. Stable Diffusion 系列（开源权重）")
    sd_models = pd.DataFrame({
        "模型": ["SDXL 1.0", "SD3 (Stable Diffusion 3)", "SD3.5 Medium/Large"],
        "发布时间": ["2023 年 9 月", "2024 年 2 月", "2025 年中"],
        "分辨率": ["1024×1024", "1024×1024", "1024×1024+"],
        "特点": [
            "- 支持 8GB 显存运行\n- 色彩、构图、手部结构显著优化\n- 包含 Base + Refiner 双阶段",
            "- 基于 **MMDiT 架构**（多模态扩散 Transformer）\n- 文本渲染能力飞跃（可生成“Hello World”等文字）\n- 支持更复杂提示词理解",
            "- 性能超越 Midjourney v6 在部分基准\n- 支持动态分辨率生成"
        ],
        "开源状态": ["✅ GitHub / Hugging Face", "✅ 权重开源（需申请）", "✅ 开源（CreativeML OpenRAIL-M）"]
    })
    st.dataframe(sd_models, use_container_width=True, hide_index=True)
    st.caption("💡 **许可协议**：均采用 **CreativeML OpenRAIL++-M**，允许商用、修改、分发，但禁止违法用途。")

    st.markdown("### 🔒 2. 闭源 API 服务（Stability AI Generate API）")
    api_features = pd.DataFrame({
        "功能": ["文生图", "图像放大", "图像编辑", "控制生成"],
        "模型名称": ["`sd3`, `sd3-turbo`", "`upscale-fast`, `upscale-conservative`, `upscale-creative`", "`edit`, `erase`, `inpainting`, `outpainting`", "`control-lora`, `depth`, `canny`, `pose`"],
        "能力": [
            "高质量图像生成",
            "快速/保真/创意放大",
            "对象擦除、内容扩展、局部重绘",
            "结合 ControlNet 实现精准控制"
        ],
        "是否开源": ["❌（API Only）", "❌", "❌", "✅（部分 LoRA 开源）"]
    })
    st.dataframe(api_features, use_container_width=True, hide_index=True)
    st.info("🌐 通过 [Stability AI 官网 API](https://platform.stability.ai/) 调用，支持企业级 SLA。")

    st.subheader("🎥 二、视频生成模型")
    st.markdown("### ✅ Stable Video Diffusion (SVD)")
    svd_info = pd.DataFrame({
        "项目": ["发布时间", "能力", "开源状态", "API 服务"],
        "详情": [
            "2023 年底 → 2025 年升级至 **SVD 2.0**",
            "- 图像 → 视频（14–25 帧，576×1024）\n- 支持相机运动控制（平移、缩放）",
            "✅ 权重公开（Hugging Face）",
            "提供 **Image-to-Video** 接口（闭源优化版，支持更高帧率）"
        ]
    })
    st.dataframe(svd_info, use_container_width=True, hide_index=True)

    st.subheader("🔊 三、音频生成模型")
    st.markdown("### ✅ 1. Stable Audio 2.5（2025 年 9 月发布）")
    audio_25 = pd.DataFrame({
        "特性": [
            "架构",
            "生成长度",
            "情感控制",
            "风格支持",
            "音频修补",
            "上下文感知",
            "部署方式",
            "版权"
        ],
        "说明": [
            "基于 **ARC（Adversarial Relativistic-Contrastive）训练技术**",
            "最长 **3 分钟** 的完整音乐作品（引子-发展-尾声）",
            "精准响应情感提示（如“振奋人心”“舒缓宁静”）",
            "支持特定风格（如“丰富的合成器声”）",
            "上传音频 → AI 续写或补全",
            "理解已有音频上下文，自然衔接",
            "- 云端 API\n- 移动端：Stable Audio Open Small（11 秒立体声，7 秒生成）",
            "训练数据为 **授权音效库**，输出可商用"
        ]
    })
    st.dataframe(audio_25, use_container_width=True, hide_index=True)

    st.markdown("### ✅ 2. Stable Audio 2.0（2024 年 4 月）")
    st.write("- 基础版本，支持文本生成背景音乐、音效")
    st.write("- 已被 2.5 全面取代")

    st.subheader("🧊 四、3D 与多模态模型")
    st.markdown("### ✅ Stable 3D")
    stable_3d = pd.DataFrame({
        "项目": ["功能", "输入", "输出", "API 状态", "应用场景"],
        "详情": [
            "单张图像 → **glTF 3D 资产**（带纹理、网格）",
            "任意 JPG/PNG",
            "可直接用于 Unity/Unreal 的 3D 模型",
            "闭源服务（POST 请求，multipart/form-data）",
            "游戏资产快速生成、电商 3D 商品展示"
        ]
    })
    st.dataframe(stable_3d, use_container_width=True, hide_index=True)

    st.subheader("📝 五、文本与嵌入模型（较少宣传）")
    lm_info = """
    - **Stable LM 系列**（早期项目）：
      - 包括 Stable LM 2（1.6B/3B/7B/12B）
      - 开源权重，但社区活跃度低于 Llama
      - **2025 年未更新**，重心已转向多模态生成
    - **嵌入模型**：用于内部 RAG 系统，未公开发布
    """
    st.markdown(lm_info)

    st.subheader("🤝 六、企业合作与定制模型")
    partnerships = pd.DataFrame({
        "合作方": ["WPP 集团", "Amp（WPP 子公司）", "DreamStudio / Clipdrop"],
        "项目": ["品牌音频标识", "音频识别服务", "消费级产品"],
        "内容": [
            "为全球品牌定制专属音效（如开机音、广告 BGM）",
            "通过 **WPP Open 平台** 向客户开放 Stable Audio 2.5",
            "提供免费试用（SDXL、Upscale、Remove Background 等）"
        ]
    })
    st.dataframe(partnerships, use_container_width=True, hide_index=True)

    st.subheader("⚙️ 七、技术底座与平台")
    tech_stack = pd.DataFrame({
        "组件": ["训练数据", "许可协议", "部署方式", "硬件支持"],
        "说明": [
            "自建 LAION 衍生数据集 + 授权商业素材",
            "CreativeML OpenRAIL++-M（宽松商用）",
            "- 本地：GitHub + ComfyUI / WebUI\n- 云端：Stability AI API、DreamStudio、Clipdrop",
            "支持消费级 GPU（RTX 3060+），SDXL 可在 8GB 显存运行"
        ]
    })
    st.dataframe(tech_stack, use_container_width=True, hide_index=True)

    st.subheader("🌐 访问方式汇总")
    access_stab = pd.DataFrame({
        "类型": ["开源模型", "API 服务", "免费试用", "本地部署"],
        "平台": ["Hugging Face", "Stability AI Platform", "Clipdrop", "GitHub"],
        "地址": [
            "[https://huggingface.co/stabilityai](https://huggingface.co/stabilityai)",
            "[https://platform.stability.ai](https://platform.stability.ai/)",
            "[https://clipdrop.co](https://clipdrop.co/)",
            "[https://github.com/Stability-AI](https://github.com/Stability-AI)"
        ]
    })
    st.dataframe(access_stab, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：Stability AI 模型战略")
    strategy = pd.DataFrame({
        "维度": ["开源", "闭源服务", "多模态扩展", "企业落地"],
        "策略": [
            "图像、音频、视频基础模型全面开源，构建全球最大生成式 AI 社区",
            "通过 API 提供增强功能（高清、编辑、3D、长音频）实现商业化",
            "从 SD → 视频 → 音频 → 3D，打造“全感官生成”能力",
            "与 WPP 等巨头合作，切入广告、品牌、零售场景"
        ]
    })
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.warning("""
    📌 **注意**：
    - **无“Stable Diffusion 4”** 官方命名，当前最新为 **SD3.5**；
    - **Stable Audio 2.5 是音频领域重大突破**，支持完整音乐创作；
    - 所有开源模型 **可免费商用**，但 API 服务按使用量计费。
    """)
# ========================
# 🔍 Perplexity AI（新增）
# ========================
with tab_perplexity:
    st.header("🔍 Perplexity AI 模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，Perplexity AI 已从一个“对话式搜索引擎”演变为具备 **自研推理模型 + 多模型路由 + 深度网络检索 + 内容生成** 能力的 **AI 原生信息平台**。
    > 
    > **核心战略**：
    > - **不只提供答案，而是提供可验证、可溯源、可行动的认知服务。**
    """)

    st.subheader("🧠 一、自研模型：Sonar 系列（核心竞争力）")
    st.markdown("> Sonar 是 Perplexity 的 **专有检索增强生成（RAG）推理引擎**，基于 Llama 3.3 70B 精调，并融合多阶段事实校验机制。")

    sonar_models = pd.DataFrame({
        "模型": ["Sonar Online", "Sonar Reasoning", "Sonar Lite"],
        "底层架构": ["Llama 3.3 70B + 自研 RAG", "MoE 架构（激活 ~32B）", "轻量稠密模型"],
        "上下文": ["实时网络检索", ">100K tokens", "32K tokens"],
        "核心优势": [
            "- 引用源数量是同类模型 **2–3 倍**\n- 支持 **聚焦搜索**（限定域名/来源）\n- 自动生成带引用的结构化报告",
            "- 多步推理 + 工具调用\n- 支持 **代码执行、数据可视化、PDF 分析**\n- “思考链”可展开查看",
            "- 低延迟、低成本\n- 适合简单问答与移动端"
        ],
        "性能表现": [
            "在 **AI 搜索竞技场** 包揽前四名，超越 Gemini、GPT-4o",
            "SWE-Bench 排名 Top 3，科研问答准确率 **91.4%**",
            "免费用户默认使用"
        ]
    })
    st.dataframe(sonar_models, use_container_width=True, hide_index=True)
    st.info("""
    💡 **Sonar 的最大优势**：
    - **强制事实绑定**：若检索不到可靠信息，直接返回“无法确认”，而非幻觉；
    - **引用可点击**：每个句子标注来源链接，支持一键跳转；
    - **报告模式**：自动生成带目录、图表、参考文献的长文（如“2025年全球AI芯片市场分析”）。
    """)

    st.subheader("🔗 二、集成第三方模型（通过 Model Router 动态调度）")
    st.markdown("Perplexity Pro / Max 用户可在设置中选择 **底层推理模型**，系统会结合 Sonar 的检索能力进行增强：")

    third_party_models = pd.DataFrame({
        "可选模型": ["Claude 3.5 Sonnet", "GPT-4o", "Gemini Pro", "Kimi K2 Thinking"],
        "提供方": ["Anthropic", "OpenAI（via Azure）", "Google Cloud", "月之暗面（Moonshot）"],
        "特点": [
            "最强综合性能，编码/推理领先",
            "多模态全能，语音/图像支持",
            "高性价比，Google 生态整合",
            "中文理解极强，长上下文处理"
        ],
        "使用场景": [
            "技术研究、开发辅助",
            "创意生成、多模态任务",
            "快速问答、内容摘要",
            "中文研究、政策分析"
        ]
    })
    st.dataframe(third_party_models, use_container_width=True, hide_index=True)
    st.caption("""
    ⚙️ **智能路由机制**：
    Perplexity 的 **Head of Search Alexandr Yarats** 表示：“我们设计了这样一个系统：**数十个 LLM（从大到小）并行工作**，以快速且成本效益高的方式处理一个用户请求。”
    """)

    st.subheader("🌐 三、深度网络检索与内容处理技术")
    st.markdown("Perplexity 不依赖传统索引，而是构建了 **实时网络抓取 + 语义过滤 + 权威性评分** 三层架构：")

    retrieval_technologies = pd.DataFrame({
        "技术模块": ["Focused Search", "Publisher Trust Score", "PDF / 网页解析引擎", "Collections"],
        "功能": [
            "用户可指定 `site:nytimes.com` 或排除社交媒体",
            "对新闻源、学术网站、政府站点打分，优先引用高可信度内容",
            "自动提取表格、图表、参考文献，支持上传文件分析",
            "用户可保存搜索结果集，形成个人知识库"
        ]
    })
    st.dataframe(retrieval_technologies, use_container_width=True, hide_index=True)

    st.subheader("📊 四、产品分层与模型访问权限")
    product_tiers = pd.DataFrame({
        "订阅等级": ["Free", "Pro（$20/月）", "Max（$200/月）"],
        "可用模型": ["Sonar Lite", "Sonar Online + Claude 3.5 Sonnet / GPT-4o / Gemini Pro", "Sonar Reasoning + 所有第三方模型"],
        "功能限制": [
            "- 每日查询限额\n- 无文件上传\n- 无高级报告",
            "- 无限制追问\n- 文件上传（PDF/DOCX）\n- Collections",
            "- Comet AI 浏览器\n- 视频生成（5–15 个/月）\n- 优先响应队列"
        ]
    })
    st.dataframe(product_tiers, use_container_width=True, hide_index=True)
    st.info("""
    🎥 **视频生成**：2025 年 8 月上线，基于 **Google Veo 3**，输入文字生成短视频（Pro 5 个/月，Max 15 个/月）。
    """)

    st.subheader("🤝 五、战略合作与生态扩展")
    partnerships_info = """
    - **Kimi K2 Thinking 接入**（2025 年 11 月）：
        - 成为 **唯一接入 Perplexity 的国产模型**；
        - 显著提升中文长文档理解与政策分析能力。
    - **C罗合作**（2025 年 12 月）：
        - 推出 **足球知识互动专区**，由 Sonar 生成赛事分析、球员数据报告。
    - **Comet AI 浏览器**：
        - 将 Sonar 深度集成至浏览器，实现“搜索即操作”（如订机票、比价）。
    """
    st.markdown(partnerships_info)

    st.subheader("📈 六、关键数据（截至 2025 年 12 月）")
    key_stats = pd.DataFrame({
        "指标": ["月活跃用户", "估值", "年收入", "市场份额", "支持语言", "覆盖国家"],
        "数值": ["3000 万", "200 亿美元", "1 亿美元（连续三年增长 500%）", "AI 搜索领域 **6.6%**（仅次于 ChatGPT 的 60.7%）", "46 种", "238 个"]
    })
    st.dataframe(key_stats, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：Perplexity 的模型战略")
    strategy_summary = pd.DataFrame({
        "维度": ["自研核心", "开放集成", "产品定位", "商业模式"],
        "策略": [
            "Sonar 系列 —— 专为 **事实准确性 + 深度引用** 优化",
            "动态路由 GPT/Claude/Gemini/Kimi，避免供应商锁定",
            "从 “答案引擎” → “认知操作系统”（通过 Comet 浏览器）",
            "订阅 + 出版商分成（4250 万美元基金，80% 分给内容方）"
        ]
    })
    st.dataframe(strategy_summary, use_container_width=True, hide_index=True)

    st.success("""
    📌 **推荐使用**：
    - **研究/学习** → 选择 **Sonar Online + Kimi K2（中文）** 或 **Claude 3.5（英文）**；
    - **快速问答** → **Sonar Lite（免费）**；
    - **深度分析** → **Sonar Reasoning（Max 订阅）**。
    """)
# ========================
# 🇨🇳 腾讯 Tencent HY（新增）
# ========================
with tab_tencent:
    st.header("🇨🇳 腾讯 Tencent HY 大模型体系（2025 年 12 月）")

    st.markdown("""
    > 截至 **2025 年 12 月**，腾讯已全面升级其大模型体系，将原“混元（Hunyuan）”品牌正式更名为 **Tencent HY**（发音同 “Hi”），并发布 **HY 2.0 系列**，  
    > 标志着其从中文大模型向全球化 AI 基础设施的战略转型。
    > 
    > **核心路线**：**“全链路自研 + 多模态覆盖 + 场景深度集成”**  
    > 模型既支持 **API 调用**，也深度嵌入微信、QQ、广告、游戏、内容生态等内部产品。
    """)

    st.subheader("🧠 一、Tencent HY（原 Hunyuan）主干模型系列")
    st.info("""
    ✅ **2025 年 12 月重大更新**：
    - 品牌名由 “Hunyuan” → **“Tencent HY”**
    - 发布 **HY 2.0 Think / Instruct** 双子架构
    - 支持 **AI 搜索插件**（整合腾讯新闻、微信搜一搜、全网实时数据）
    """)

    hy_models = pd.DataFrame({
        "模型名称": [
            "Tencent HY 2.0 Think",
            "Tencent HY 2.0 Instruct",
            "Hunyuan-A13B",
            "Hunyuan-T1-latest",
            "Hunyuan-TurboS-latest"
        ],
        "类型": [
            "推理/思考型",
            "指令跟随型",
            "长文本专用",
            "多模态生成",
            "轻量高速"
        ],
        "最大输入": ["**128K tokens**", "128K tokens", "**224K tokens**", "32K tokens", "32K tokens"],
        "最大输出": ["**64K tokens**", "16K tokens", "32K tokens", "**64K tokens**", "16K tokens"],
        "特点": [
            "强逻辑推理、多步规划、代码生成",
            "高效执行指令、内容创作、润色",
            "“大海捞针”准确率 **99.9%**",
            "图文理解与生成",
            "低延迟、高并发"
        ],
        "适用场景": [
            "科研、智能体、复杂问答",
            "客服、营销、办公自动化",
            "法律、金融长文档处理",
            "社交媒体、内容平台",
            "实时对话、小程序集成"
        ]
    })
    st.dataframe(hy_models, use_container_width=True, hide_index=True)
    st.success("""
    💡 **优势**：
    - 中文理解与生成能力 **业界领先**（CLUE、C-Eval 排名第一）
    - 支持 **万亿级参数 MoE 架构**，推理成本下降 40%
    - 内置 **安全防护体系**（数据脱敏、内容审核、权限控制）
    """)

    st.subheader("🖼️ 二、多模态与专用模型")
    multimodal_models = pd.DataFrame({
        "模型名称": [
            "Hunyuan-Translation",
            "Hunyuan-Translation-lite",
            "Hunyuan-Vision",
            "Hunyuan-Audio",
            "Hunyuan-Code"
        ],
        "功能": [
            "高精度翻译",
            "轻量翻译",
            "图像理解",
            "语音识别与合成",
            "代码生成"
        ],
        "状态": [
            "输入/输出 4K，支持 100+ 语种",
            "同上，更低延迟",
            "支持 OCR、图表解析、商品识别",
            "微信语音消息转写、虚拟主播",
            "支持微信小程序、云开发、游戏脚本"
        ]
    })
    st.dataframe(multimodal_models, use_container_width=True, hide_index=True)
    st.caption("🔒 所有多模态模型均通过 **腾讯云 TI 平台** 提供企业级 API。")

    st.subheader("🤖 三、独立 AI 产品与智能体")
    ai_products = pd.DataFrame({
        "产品": ["元宝（Yuanbao）", "微信 AI 功能", "QQ 小世界 AI 创作"],
        "说明": [
            "腾讯官方 AI 助手（网页/APP）",
            "公众号摘要、朋友圈文案、视频字幕",
            "自动配文、表情包生成"
        ],
        "是否受更名影响": [
            "❌ **不受影响**，仍可正常使用\n底层调用 Tencent HY 模型",
            "深度集成 HY 模型",
            "使用 Hunyuan-T1 系列"
        ]
    })
    st.dataframe(ai_products, use_container_width=True, hide_index=True)
    st.info("✅ 用户可通过 **元宝** 免费体验 Tencent HY 能力，无需技术门槛。")

    st.subheader("⚙️ 四、技术特性与基础设施")
    tech_features = pd.DataFrame({
        "能力": ["上下文长度", "知识增强", "安全合规", "部署方式"],
        "说明": [
            "最高支持 **256K tokens**（部分内部版本）",
            "实时联网 + 腾讯内容生态（新闻、视频、百科）",
            "全流程数据加密、GDPR/CCPA 合规、国产化适配",
            "- 公有云 API（腾讯云）\n- 私有化部署（金融/政务专属）\n- SaaS 服务（如文档助手、表格公式生成）"
        ]
    })
    st.dataframe(tech_features, use_container_width=True, hide_index=True)

    st.subheader("🌐 访问与使用方式")
    access_info = pd.DataFrame({
        "平台": [
            "腾讯混元大模型官网",
            "腾讯云控制台",
            "元宝（Yuanbao）",
            "新用户资源包"
        ],
        "地址": [
            "[https://hunyuan.tencent.com](https://hunyuan.tencent.com/)",
            "[https://console.cloud.tencent.com/hunyuan](https://console.cloud.tencent.com/hunyuan)",
            "[https://yuanbao.tencent.com](https://yuanbao.tencent.com/)",
            "控制台领取"
        ],
        "用途": [
            "文档、控制台、申请试用",
            "购买 API、管理资源",
            "免费对话体验",
            "- 文本模型：**100 万 tokens**\n- Embedding 模型：**100 万 tokens**"
        ]
    })
    st.dataframe(access_info, use_container_width=True, hide_index=True)

    st.subheader("✅ 总结：腾讯大模型战略亮点")
    strategy_highlights = """
    - **品牌升级**：从 “Hunyuan” → **“Tencent HY”**，强化国际化形象；
    - **双轨架构**：**Think（重推理） + Instruct（重执行）** 满足不同场景；
    - **生态融合**：深度集成微信、QQ、广告、游戏、内容平台，实现“模型即服务”；
    - **安全可靠**：全流程安全防护，适合金融、政务等高合规要求场景；
    - **免费体验**：新用户送 200 万 tokens，元宝 APP 零门槛使用。
    """
    st.markdown(strategy_highlights)

    st.warning("""
    📌 **注意**：
    - “混元”作为技术代号仍在内部使用，但对外统一称 **Tencent HY**；
    - **HY 3.0 已在研发中**，预计 2026 年 Q2 发布，将支持 **视频生成与具身智能**。
    """)
# ======================
# 页脚
# ======================
st.divider()
st.caption("数据截至 2025 年 12 月 | 来源：各公司官网、文档及公开资料整理")