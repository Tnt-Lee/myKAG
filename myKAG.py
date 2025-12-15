#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动知识图谱构建系统
支持技术文档、保险业务文档的实体识别和关系抽取
支持中文、日语、英语文档
"""

import spacy
import json
import re
from typing import List, Dict, Tuple, Set
from neo4j import GraphDatabase
from collections import defaultdict, Counter
import logging
from pathlib import Path
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityRelationExtractor:
    """实体关系抽取器"""
    
    def __init__(self, languages=['zh', 'ja', 'en']):
        """
        初始化多语言NLP模型
        
        Args:
            languages: 支持的语言列表 ['zh', 'ja', 'en']
        """
        self.nlp_models = {}
        self.load_nlp_models(languages)
        
        # 定义关系模式（基于依存句法分析）
        self.relation_patterns = {
            # 英语关系模式
            'en': [
                # 主谓宾关系
                {'pattern': r'nsubj.*rel.*dobj', 'relation': 'ACTS_ON'},
                {'pattern': r'nsubj.*cop.*attr', 'relation': 'IS_A'},
                {'pattern': r'compound.*head', 'relation': 'PART_OF'},
                {'pattern': r'prep.*pobj', 'relation': 'RELATED_TO'},
                {'pattern': r'appos', 'relation': 'ALSO_KNOWN_AS'},
            ],
            # 中文关系模式
            'zh': [
                {'pattern': r'nsubj.*root.*dobj', 'relation': 'ACTS_ON'},
                {'pattern': r'nsubj.*cop.*attr', 'relation': 'IS_A'},
                {'pattern': r'compound.*head', 'relation': 'PART_OF'},
                {'pattern': r'prep.*pobj', 'relation': 'RELATED_TO'},
            ],
            # 日语关系模式
            'ja': [
                {'pattern': r'nsubj.*root.*dobj', 'relation': 'ACTS_ON'},
                {'pattern': r'compound.*head', 'relation': 'PART_OF'},
                {'pattern': r'prep.*pobj', 'relation': 'RELATED_TO'},
            ]
        }
    
    def load_nlp_models(self, languages):
        """加载多语言NLP模型"""
        model_map = {
            'en': 'en_core_web_sm',
            'zh': 'zh_core_web_sm', 
            'ja': 'ja_core_news_sm'
        }
        
        for lang in languages:
            try:
                model_name = model_map[lang]
                self.nlp_models[lang] = spacy.load(model_name)
                logger.info(f"已加载 {lang} 语言模型: {model_name}")
            except IOError:
                logger.warning(f"未找到 {lang} 语言模型 {model_map[lang]}，请安装: python -m spacy download {model_map[lang]}")
                # 如果无法加载特定语言模型，尝试使用空白模型
                try:
                    self.nlp_models[lang] = spacy.blank(lang)
                    logger.info(f"使用 {lang} 空白模型作为备用")
                except Exception as e:
                    logger.error(f"无法为 {lang} 创建任何模型: {e}")
                    self.nlp_models[lang] = None  # 设置为None而不是跳过
                    continue
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测逻辑
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        else:
            return 'en'
    
    def extract_entities(self, text: str) -> List[Dict]:
        """提取实体"""
        if not text or not text.strip():
            return []
            
        lang = self.detect_language(text)
        if lang not in self.nlp_models or self.nlp_models[lang] is None:
            lang = 'en'  # 默认使用英语
            if lang not in self.nlp_models or self.nlp_models[lang] is None:
                logger.warning("没有可用的NLP模型，使用基础文本处理")
                return self._basic_entity_extraction(text)
        
        nlp = self.nlp_models[lang]
        
        try:
            doc = nlp(text)
        except Exception as e:
            logger.error(f"NLP处理失败: {e}")
            return self._basic_entity_extraction(text)
        
        entities = []
        
        # 命名实体识别
        try:
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'type': 'NAMED_ENTITY'
                })
        except Exception as e:
            logger.warning(f"命名实体识别失败: {e}")
        
        # 名词短语提取（根据语言选择不同策略）
        try:
            if lang == 'en':
                # 英语使用noun_chunks
                if hasattr(doc, 'noun_chunks'):
                    for chunk in doc.noun_chunks:
                        if len(chunk.text.strip()) > 1:
                            entities.append({
                                'text': chunk.text,
                                'label': 'NOUN_PHRASE',
                                'start': chunk.start_char,
                                'end': chunk.end_char,
                                'type': 'CONCEPT'
                            })
                else:
                    entities.extend(self._extract_nouns_by_pos(doc, lang))
            else:
                # 中文和日语使用基于词性的名词提取
                entities.extend(self._extract_nouns_by_pos(doc, lang))
        except Exception as e:
            logger.warning(f"名词短语提取失败: {e}")
        
        # 去重并按重要性排序
        unique_entities = self._deduplicate_entities(entities)
        return unique_entities if unique_entities else []
    
    def _extract_nouns_by_pos(self, doc, lang: str) -> List[Dict]:
        """基于词性标注提取名词（用于中文和日语）"""
        entities = []
        
        # 定义名词相关的词性标签
        noun_pos_tags = {
            'zh': ['NOUN', 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS'],  # 中文名词标签
            'ja': ['NOUN', 'PROPN', '名詞']  # 日语名词标签
        }
        
        target_pos = noun_pos_tags.get(lang, ['NOUN', 'PROPN'])
        
        # 提取连续的名词序列
        current_noun_phrase = []
        start_char = None
        
        for token in doc:
            # 跳过标点符号和停用词
            if token.is_punct or token.is_stop or len(token.text.strip()) == 0:
                if current_noun_phrase:
                    # 结束当前名词短语
                    phrase_text = ''.join([t.text for t in current_noun_phrase])
                    if len(phrase_text.strip()) > 1:
                        entities.append({
                            'text': phrase_text,
                            'label': 'NOUN_PHRASE',
                            'start': start_char,
                            'end': current_noun_phrase[-1].idx + len(current_noun_phrase[-1].text),
                            'type': 'CONCEPT'
                        })
                    current_noun_phrase = []
                    start_char = None
                continue
            
            # 检查是否为名词
            if token.pos_ in target_pos or any(pos in token.tag_ for pos in target_pos):
                if not current_noun_phrase:
                    start_char = token.idx
                current_noun_phrase.append(token)
            else:
                # 非名词，结束当前短语
                if current_noun_phrase:
                    phrase_text = ''.join([t.text for t in current_noun_phrase])
                    if len(phrase_text.strip()) > 1:
                        entities.append({
                            'text': phrase_text,
                            'label': 'NOUN_PHRASE',
                            'start': start_char,
                            'end': current_noun_phrase[-1].idx + len(current_noun_phrase[-1].text),
                            'type': 'CONCEPT'
                        })
                    current_noun_phrase = []
                    start_char = None
        
        # 处理文档末尾的名词短语
        if current_noun_phrase:
            phrase_text = ''.join([t.text for t in current_noun_phrase])
            if len(phrase_text.strip()) > 1:
                entities.append({
                    'text': phrase_text,
                    'label': 'NOUN_PHRASE',
                    'start': start_char,
                    'end': current_noun_phrase[-1].idx + len(current_noun_phrase[-1].text),
                    'type': 'CONCEPT'
                })
        
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = set()
        unique = []
        
        # 按长度降序排列，优先保留长实体
        entities.sort(key=lambda x: len(x['text']), reverse=True)
        
        for entity in entities:
            text = entity['text'].lower().strip()
            if text not in seen and len(text) > 1:
                seen.add(text)
                unique.append(entity)
        
        return unique
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """提取实体间关系"""
        if not entities:  # 如果没有实体，返回空列表
            return []
            
        lang = self.detect_language(text)
        if lang not in self.nlp_models:
            lang = 'en'
        
        # 如果没有可用的NLP模型，只使用共现关系
        if lang not in self.nlp_models or self.nlp_models[lang] is None:
            logger.warning(f"没有可用的 {lang} 模型，只使用共现关系")
            return self._extract_cooccurrence_relations(text, entities)
        
        nlp = self.nlp_models[lang]
        doc = nlp(text)
        
        relations = []
        
        # 基于依存句法分析的关系抽取
        try:
            for sent in doc.sents:
                sent_relations = self._extract_sentence_relations(sent, entities, lang)
                if sent_relations:  # 检查是否为None
                    relations.extend(sent_relations)
        except Exception as e:
            logger.warning(f"依存句法分析失败: {e}")
        
        # 基于共现的关系抽取
        try:
            cooccurrence_relations = self._extract_cooccurrence_relations(text, entities)
            if cooccurrence_relations:  # 检查是否为None
                relations.extend(cooccurrence_relations)
        except Exception as e:
            logger.warning(f"共现关系抽取失败: {e}")
        
        return relations if relations else []
    
    def _extract_sentence_relations(self, sent, entities: List[Dict], lang: str) -> List[Dict]:
        """从单个句子中提取关系"""
        relations = []
        
        if not sent or not entities:
            return relations
        
        # 获取句子中的实体
        sent_entities = []
        for entity in entities:
            if entity and 'start' in entity and 'end' in entity:
                if entity['start'] >= sent.start_char and entity['end'] <= sent.end_char:
                    sent_entities.append(entity)
        
        if len(sent_entities) < 2:
            return relations
        
        # 基于依存关系提取
        try:
            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass'] and token.head.pos_ == 'VERB':
                    subj_entity = self._find_entity_by_token(token, sent_entities)
                    
                    # 查找宾语
                    for child in token.head.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            obj_entity = self._find_entity_by_token(child, sent_entities)
                            
                            if subj_entity and obj_entity:
                                relations.append({
                                    'source': subj_entity['text'],
                                    'target': obj_entity['text'],
                                    'relation': self._determine_relation_type(token.head.text, lang),
                                    'confidence': 0.8
                                })
        except Exception as e:
            logger.warning(f"句法分析关系提取失败: {e}")
        
        return relations
    
    def _extract_cooccurrence_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于共现关系提取"""
        relations = []
        
        if not text or not entities:
            return relations
        
        # 在同一段落中的实体建立RELATED_TO关系
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            para_entities = []
            for entity in entities:
                if entity and 'text' in entity and entity['text'] in para:
                    para_entities.append(entity)
            
            # 两两建立关系
            for i in range(len(para_entities)):
                for j in range(i + 1, len(para_entities)):
                    if para_entities[i] and para_entities[j]:
                        relations.append({
                            'source': para_entities[i]['text'],
                            'target': para_entities[j]['text'],
                            'relation': 'RELATED_TO',
                            'confidence': 0.5
                        })
        
        return relations
    
    def _find_entity_by_token(self, token, entities: List[Dict]):
        """根据token查找对应的实体"""
        if not token or not entities:
            return None
            
        for entity in entities:
            if entity and 'text' in entity and token.text in entity['text']:
                return entity
        return None
    
    def _determine_relation_type(self, verb: str, lang: str) -> str:
        """根据动词确定关系类型"""
        # 简化的关系类型映射
        relation_map = {
            'en': {
                'is': 'IS_A', 'are': 'IS_A', 'was': 'IS_A', 'were': 'IS_A',
                'has': 'HAS', 'have': 'HAS', 'had': 'HAS',
                'use': 'USES', 'uses': 'USES', 'used': 'USES',
                'contain': 'CONTAINS', 'contains': 'CONTAINS',
                'provide': 'PROVIDES', 'provides': 'PROVIDES'
            },
            'zh': {
                '是': 'IS_A', '为': 'IS_A',
                '有': 'HAS', '具有': 'HAS',
                '使用': 'USES', '采用': 'USES',
                '包含': 'CONTAINS', '包括': 'CONTAINS',
                '提供': 'PROVIDES'
            },
            'ja': {
                'である': 'IS_A', 'です': 'IS_A',
                'を持つ': 'HAS', 'がある': 'HAS',
                'を使用': 'USES', 'を利用': 'USES',
                'を含む': 'CONTAINS',
                'を提供': 'PROVIDES'
            }
        }
        
        return relation_map.get(lang, {}).get(verb.lower(), 'RELATED_TO')


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱操作类"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.verify_connectivity()
    
    def verify_connectivity(self):
        """验证数据库连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def create_entity(self, entity: Dict) -> None:
        """创建实体节点"""
        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {name: $name, type: $type})
            SET e.label = $label
            RETURN e
            """
            session.run(query, 
                       name=entity['text'], 
                       type=entity.get('type', 'UNKNOWN'),
                       label=entity.get('label', ''))
    
    def create_relation(self, relation: Dict) -> None:
        """创建关系"""
        with self.driver.session() as session:
            query = """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            MERGE (a)-[r:RELATION {type: $relation_type, confidence: $confidence}]->(b)
            RETURN r
            """
            session.run(query,
                       source=relation['source'],
                       target=relation['target'],
                       relation_type=relation['relation'],
                       confidence=relation.get('confidence', 0.5))
    
    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("已清空Neo4j数据库")
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        with self.driver.session() as session:
            # 节点数量
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            
            # 关系数量
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            # 节点类型分布
            node_types = session.run("""
                MATCH (n:Entity) 
                RETURN n.type as type, count(n) as count 
                ORDER BY count DESC
            """).data()
            
            return {
                'total_nodes': node_count,
                'total_relations': rel_count,
                'node_types': node_types
            }
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()


class KnowledgeGraphBuilder:
    """知识图谱构建器主类"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        初始化知识图谱构建器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: 用户名  
            neo4j_password: 密码
        """
        self.extractor = EntityRelationExtractor()
        self.graph_db = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        
    def build_from_text(self, text: str, clear_existing: bool = False) -> Dict:
        """
        从文本构建知识图谱
        
        Args:
            text: 输入文本
            clear_existing: 是否清空现有数据
            
        Returns:
            构建统计信息
        """
        if clear_existing:
            self.graph_db.clear_database()
        
        logger.info("开始提取实体...")
        entities = self.extractor.extract_entities(text)
        logger.info(f"提取到 {len(entities)} 个实体")
        
        logger.info("开始提取关系...")
        relations = self.extractor.extract_relations(text, entities)
        logger.info(f"提取到 {len(relations)} 个关系")
        
        # 创建实体节点
        logger.info("创建实体节点...")
        for entity in entities:
            self.graph_db.create_entity(entity)
        
        # 创建关系
        logger.info("创建关系...")
        for relation in relations:
            self.graph_db.create_relation(relation)
        
        stats = self.graph_db.get_statistics()
        logger.info(f"知识图谱构建完成: {stats}")
        
        return {
            'extracted_entities': len(entities),
            'extracted_relations': len(relations),
            'graph_stats': stats
        }
    
    def build_from_file(self, file_path: str, clear_existing: bool = False) -> Dict:
        """
        从文件构建知识图谱
        
        Args:
            file_path: 文件路径
            clear_existing: 是否清空现有数据
            
        Returns:
            构建统计信息
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择读取方式
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.suffix.lower() in ['.csv']:
            df = pd.read_csv(file_path)
            text = df.to_string()
        elif file_path.suffix.lower() in ['.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        return self.build_from_text(text, clear_existing)
    
    def export_graph_data(self, output_file: str = 'knowledge_graph.json'):
        """导出图谱数据"""
        with self.graph_db.driver.session() as session:
            # 导出节点
            nodes_result = session.run("""
                MATCH (n:Entity) 
                RETURN n.name as name, n.type as type, n.label as label
            """)
            nodes = [dict(record) for record in nodes_result]
            
            # 导出关系
            relations_result = session.run("""
                MATCH (a:Entity)-[r:RELATION]->(b:Entity) 
                RETURN a.name as source, b.name as target, 
                       r.type as relation, r.confidence as confidence
            """)
            relations = [dict(record) for record in relations_result]
            
            # 保存到文件
            graph_data = {
                'nodes': nodes,
                'relations': relations,
                'statistics': self.graph_db.get_statistics()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"图谱数据已导出到: {output_file}")
            return graph_data
    
    def close(self):
        """关闭所有连接"""
        self.graph_db.close()


def main():
    """主函数示例"""
    # 配置Neo4j连接
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"  
    NEO4J_PASSWORD = "password"
    
    # 示例文档内容
    sample_texts = {
        'insurance_zh': """
        保险合同是投保人与保险人约定保险权利义务关系的协议。
        投保人向保险人支付保险费，保险人对于合同约定的可能发生的事故因其发生所造成的财产损失承担赔偿保险金责任。
        人身保险包括人寿保险、健康保险和意外伤害保险。
        财产保险包括机动车保险、企业财产保险和家庭财产保险。
        保险公司通过风险评估来确定保险费率。
        """,
        
        'technical_en': """
        A microservice architecture is a method of developing software systems that are loosely coupled and independently deployable.
        Each microservice runs in its own process and communicates with other services through well-defined APIs.
        Docker containers provide lightweight virtualization for deploying microservices.
        Kubernetes orchestrates containerized applications across clusters.
        API Gateway manages and routes requests to appropriate microservices.
        """,
        
        'business_ja': """
        クラウドコンピューティングは、インターネットを通じてコンピューティングリソースを提供するサービスです。
        Amazon Web Servicesは主要なクラウドプロバイダーの一つです。
        Infrastructure as a Service（IaaS）は仮想マシンとストレージを提供します。
        Platform as a Service（PaaS）は開発プラットフォームを提供します。
        Software as a Service（SaaS）は完全なアプリケーションを提供します。
        """
    }
    
    try:
        # 检查可用的语言模型
        available_languages = []
        for lang in ['zh', 'ja', 'en']:
            try:
                if lang == 'en':
                    spacy.load('en_core_web_sm')
                elif lang == 'zh':
                    spacy.load('zh_core_web_sm')
                elif lang == 'ja':
                    spacy.load('ja_core_news_sm')
                available_languages.append(lang)
            except:
                logger.warning(f"语言模型 {lang} 不可用，将跳过或使用备用方案")
                continue
        
        if not available_languages:
            logger.error("没有可用的语言模型，请至少安装一个spaCy模型")
            return
        
        # 初始化知识图谱构建器
        kg_builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # 处理多语言文档
        for doc_type, text in sample_texts.items():
            logger.info(f"处理 {doc_type} 文档...")
            try:
                result = kg_builder.build_from_text(text, clear_existing=(doc_type == 'insurance_zh'))
                print(f"{doc_type} 处理结果: {result}")
            except Exception as e:
                logger.error(f"处理 {doc_type} 时出错: {e}")
                continue
        
        # 导出图谱数据
        kg_builder.export_graph_data('multi_language_kg.json')
        
        # 显示最终统计
        final_stats = kg_builder.graph_db.get_statistics()
        print(f"\n最终图谱统计: {final_stats}")
        
    except Exception as e:
        logger.error(f"构建过程出错: {e}")
    finally:
        if 'kg_builder' in locals():
            kg_builder.close()


if __name__ == "__main__":
    main()