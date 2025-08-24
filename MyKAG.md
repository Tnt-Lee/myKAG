##### 
    创建一个虚拟环境： python -m venv mykag_env
    在Windows上激活虚拟环境：  mykag_env\Scripts\activate
    安装依赖包  ： pip install -r requirements.txt

    更新pip版本
    python.exe -m pip install --upgrade pip

    Miniconda 创建指定 Python 版本虚拟环境
    conda create --name mykag_env python=3.12
    命令激活虚拟环境
    conda activate myenv
    退出虚拟环境
    conda deactivate

    虚拟环境的存储目录
    %USERPROFILE%\Miniconda3\envs\

    查看环境路径
    conda info --envs

# 安装Python包
pip install spacy neo4j pandas

# 下载语言模型
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm  
python -m spacy download ja_core_news_sm


# 配置数据库连接
kg_builder = KnowledgeGraphBuilder("bolt://localhost:7687", "neo4j", "password")

# 从文本构建
result = kg_builder.build_from_text(your_text)

# 从文件构建  
result = kg_builder.build_from_file("document.txt")

# 导出数据
kg_builder.export_graph_data("output.json")