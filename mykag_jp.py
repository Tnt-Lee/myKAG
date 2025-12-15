#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语保险业务文档知识图谱构建系统
支持多种日语处理方式：SudachiPy, Janome, spaCy, 在线API
无需MeCab依赖
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
from neo4j import GraphDatabase

# 配置日志
logging.basicConfig(level=logging.INFO, filename='mykag002.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体类"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class Relation:
    """关系类"""
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0

class JapaneseTokenizer:
    """日语分词器基类"""
    
    def tokenize(self, text: str) -> List[Dict]:
        """分词，返回包含词汇信息的字典列表"""
        raise NotImplementedError

class SudachiTokenizer(JapaneseTokenizer):
    """SudachiPy分词器 (推荐替代方案1)"""
    
    def __init__(self):
        try:
            from sudachipy import tokenizer
            from sudachipy import dictionary
            self.tokenizer_obj = dictionary.Dictionary().create()
            self.mode = tokenizer.Tokenizer.SplitMode.C
            self.available = True
            logger.info("SudachiPy初始化成功")
        except ImportError:
            self.available = False
            logger.warning("SudachiPy未安装")
    
    def tokenize(self, text: str) -> List[Dict]:
        if not self.available:
            return []
        
        tokens = []
        try:
            morphemes = self.tokenizer_obj.tokenize(text, self.mode)
            for m in morphemes:
                tokens.append({
                    'surface': m.surface(),
                    'reading': m.reading_form(),
                    'pos': m.part_of_speech()[0],
                    'pos_detail': '-'.join(m.part_of_speech()),
                    'base_form': m.dictionary_form()
                })
        except Exception as e:
            logger.error(f"SudachiPy分词错误: {e}")
        
        return tokens

class JanomeTokenizer(JapaneseTokenizer):
    """Janome分词器 (替代方案2)"""
    
    def __init__(self):
        try:
            from janome.tokenizer import Tokenizer
            self.tokenizer = Tokenizer()
            self.available = True
            logger.info("Janome初始化成功")
        except ImportError:
            self.available = False
            logger.warning("Janome未安装")
    
    def tokenize(self, text: str) -> List[Dict]:
        if not self.available:
            return []
        
        tokens = []
        try:
            for token in self.tokenizer.tokenize(text):
                tokens.append({
                    'surface': token.surface,
                    'reading': token.reading if hasattr(token, 'reading') else '',
                    'pos': token.pos.split(',')[0],
                    'pos_detail': token.pos,
                    'base_form': token.base_form
                })
        except Exception as e:
            logger.error(f"Janome分词错误: {e}")
        
        return tokens

class SpacyTokenizer(JapaneseTokenizer):
    """spaCy分词器 (替代方案3)"""
    
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("ja_core_news_sm")
            self.available = True
            logger.info("spaCy日语模型初始化成功")
        except (ImportError, OSError):
            try:
                import spacy
                # 尝试加载更小的模型
                self.nlp = spacy.load("ja_core_news_lg")
                self.available = True
                logger.info("spaCy日语大模型初始化成功")
            except (ImportError, OSError):
                self.available = False
                logger.warning("spaCy日语模型未安装")
    
    def tokenize(self, text: str) -> List[Dict]:
        if not self.available:
            return []
        
        tokens = []
        try:
            doc = self.nlp(text)
            for token in doc:
                tokens.append({
                    'surface': token.text,
                    'reading': token.lemma_,
                    'pos': token.pos_,
                    'pos_detail': f"{token.pos_}-{token.tag_}",
                    'base_form': token.lemma_
                })
        except Exception as e:
            logger.error(f"spaCy分词错误: {e}")
        
        return tokens

class RegexTokenizer(JapaneseTokenizer):
    """基于正则表达式的简单分词器 (最后备选方案)"""
    
    def __init__(self):
        self.available = True
        # 日语字符正则表达式
        self.hiragana = r'[\u3040-\u309F]'
        self.katakana = r'[\u30A0-\u30FF]'
        self.kanji = r'[\u4E00-\u9FAF]'
        self.ascii_chars = r'[a-zA-Z0-9]'
        self.numbers = r'[\d]'
        self.punctuation = r'[。、！？（）「」『』【】〈〉《》]'
        
        logger.info("正则表达式分词器初始化成功")
    
    def tokenize(self, text: str) -> List[Dict]:
        tokens = []
        
        # 简单的日语分词规则
        pattern = f'({self.kanji}+|{self.hiragana}+|{self.katakana}+|{self.ascii_chars}+|{self.numbers}+|{self.punctuation})'
        
        try:
            matches = re.finditer(pattern, text)
            for match in matches:
                surface = match.group()
                if surface.strip():  # 跳过空白
                    tokens.append({
                        'surface': surface,
                        'reading': surface,
                        'pos': self._guess_pos(surface),
                        'pos_detail': self._guess_pos(surface),
                        'base_form': surface
                    })
        except Exception as e:
            logger.error(f"正则表达式分词错误: {e}")
        
        return tokens
    
    def _guess_pos(self, surface: str) -> str:
        """简单的词性推测"""
        if re.match(r'\d', surface):
            return '数'
        elif re.match(self.punctuation, surface):
            return '記号'
        elif re.match(f'^{self.hiragana}+$', surface):
            return '助詞'  # 大多数平假名是助词
        elif re.match(f'^{self.katakana}+$', surface):
            return '名詞'  # 大多数片假名是名词
        elif re.match(f'^{self.kanji}+$', surface):
            return '名詞'  # 大多数汉字是名词
        else:
            return '名詞'

class JapaneseNLPProcessor:
    """日语自然语言处理器"""
    
    def __init__(self, prefer_tokenizer: str = 'auto'):
        # 尝试初始化可用的分词器
        self.tokenizers = {
            'sudachi': SudachiTokenizer(),
            'janome': JanomeTokenizer(),
            'spacy': SpacyTokenizer(),
            'regex': RegexTokenizer()
        }
        
        # 选择分词器
        if prefer_tokenizer == 'auto':
            self.tokenizer = self._select_best_tokenizer()
        else:
            self.tokenizer = self.tokenizers.get(prefer_tokenizer)
            if not self.tokenizer or not self.tokenizer.available:
                logger.warning(f"指定的分词器 {prefer_tokenizer} 不可用，自动选择")
                self.tokenizer = self._select_best_tokenizer()
        
        # 保险业务相关的实体类型定义
        self.entity_patterns = {
            'INSURANCE_PRODUCT': [
                r'生命保険|損害保険|医療保険|がん保険|学資保険|年金保険|介護保険|終身保険',
                r'自動車保険|火災保険|地震保険|旅行保険|ペット保険|傷害保険'
            ],
            'PERSON': [
                r'契約者|被保険者|受益者|保険金受取人|代理店|営業担当者|顧客|お客様',
                r'医師|弁護士|税理士|ファイナンシャルプランナー|[一-龯]{2,4}様?'
            ],
            'ORGANIZATION': [
                r'保険会社|代理店|病院|銀行|証券会社|信託銀行',
                r'株式会社|有限会社|合同会社|相互会社|[一-龯]{2,10}生命|[一-龯]{2,10}保険'
            ],
            'MONEY': [
                r'\d+円|\d+万円|\d+億円|保険料|保険金|給付金|解約返戻金|配当金',
                r'年間保険料|月額保険料|一時払保険料|日額\d+円'
            ],
            'DATE': [
                r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|令和\d+年\d{1,2}月\d{1,2}日',
                r'契約日|満期日|更新日|解約日|支払日|平成\d+年|昭和\d+年'
            ],
            'DOCUMENT': [
                r'保険証券|約款|パンフレット|申込書|診断書|請求書|保険設計書',
                r'契約書|覚書|同意書|委任状|証券番号|契約番号'
            ],
            'COVERAGE': [
                r'死亡保障|医療保障|がん保障|介護保障|障害保障|災害保障',
                r'入院給付|手術給付|通院給付|診断給付|特約'
            ],
            'INSURANCE_NUMBER': [
                r'[A-Z]+-\d{4}-\d+|証券番号[:：]\s*[A-Z0-9-]+|契約番号[:：]\s*[A-Z0-9-]+'
            ]
        }
        
        # 関係パターンの定義
        self.relation_patterns = {
            'CONTRACTED': ['契約', '加入', '申込', '締結'],
            'PAYS': ['支払', '給付', '受取', '払込'],
            'COVERS': ['保障', 'カバー', '対象', '補償'],
            'MANAGED_BY': ['管理', '担当', '代理', '取扱'],
            'ISSUED_BY': ['発行', '作成', '提供', '交付'],
            'VALID_UNTIL': ['満期', '期限', '有効', '終了'],
            'BENEFICIARY_OF': ['受益者', '受取人', '指定'],
            'RELATED_TO': ['関連', '付随', '含む', 'について']
        }
    
    def _select_best_tokenizer(self) -> JapaneseTokenizer:
        """最適な分词器を選択"""
        # 優先順位: SudachiPy > Janome > spaCy > Regex
        for name in ['sudachi', 'janome', 'spacy', 'regex']:
            tokenizer = self.tokenizers[name]
            if tokenizer.available:
                logger.info(f"選択された分词器: {name}")
                return tokenizer
        
        # 最悪の場合はRegexを返す
        return self.tokenizers['regex']
    
    def tokenize(self, text: str) -> List[Dict]:
        """テキストを分词"""
        return self.tokenizer.tokenize(text)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """実体抽出"""
        entities = []
        
        # パターンベースの実体抽出
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8
                    )
                    entities.append(entity)
        
        # 分词結果を使った実体抽出
        tokens = self.tokenize(text)
        current_pos = 0
        
        for token in tokens:
            surface = token['surface']
            pos = token['pos']
            
            # 固有名詞の抽出
            if '名詞' in pos and len(surface) > 1:
                start_pos = text.find(surface, current_pos)
                if start_pos != -1:
                    # 保険用語かチェック
                    if self._is_insurance_term(surface):
                        entity = Entity(
                            text=surface,
                            label='INSURANCE_TERM',
                            start=start_pos,
                            end=start_pos + len(surface),
                            confidence=0.7
                        )
                        entities.append(entity)
                    current_pos = start_pos + len(surface)
        
        # 重複除去とソート
        entities = self._remove_duplicate_entities(entities)
        entities.sort(key=lambda x: x.start)
        
        return entities
    
    def _is_insurance_term(self, term: str) -> bool:
        """保険用語かどうか判定"""
        insurance_keywords = [
            '保険', '契約', '保障', '給付', '特約', '更新', '満期', '解約',
            '払込', '受益', '被保険', '診査', '査定', '支払', '免責'
        ]
        return any(keyword in term for keyword in insurance_keywords)
    
    def _remove_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """重複する実体を除去"""
        unique_entities = []
        seen = set()
        
        for entity in entities:
            # 位置ベースの重複チェック（同じ位置の長いエンティティを優先）
            overlap_found = False
            for existing in unique_entities:
                if (entity.start >= existing.start and entity.start < existing.end) or \
                   (entity.end > existing.start and entity.end <= existing.end):
                    overlap_found = True
                    # より長いエンティティを保持
                    if len(entity.text) > len(existing.text):
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    break
            
            if not overlap_found:
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """関係抽出"""
        relations = []
        
        # 文を分割
        sentences = re.split(r'[。！？\n]', text)
        
        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue
            
            # 文中の実体を特定
            sentence_entities = []
            for entity in entities:
                if entity.text in sentence:
                    sentence_entities.append(entity)
            
            # 実体間の関係を抽出
            for i, ent1 in enumerate(sentence_entities):
                for j, ent2 in enumerate(sentence_entities):
                    if i != j:
                        relation_type = self._identify_relation(sentence, ent1, ent2)
                        if relation_type:
                            relation = Relation(
                                source=ent1.text,
                                target=ent2.text,
                                relation_type=relation_type,
                                confidence=0.7
                            )
                            relations.append(relation)
        
        return relations
    
    def _identify_relation(self, sentence: str, ent1: Entity, ent2: Entity) -> Optional[str]:
        """実体間の関係を特定"""
        # 実体間のテキストを取得
        pos1 = sentence.find(ent1.text)
        pos2 = sentence.find(ent2.text)
        
        if pos1 == -1 or pos2 == -1:
            return None
        
        start_pos = min(pos1, pos1 + len(ent1.text), pos2, pos2 + len(ent2.text))
        end_pos = max(pos1, pos1 + len(ent1.text), pos2, pos2 + len(ent2.text))
        
        between_text = sentence[start_pos:end_pos]
        
        # 関係パターンとのマッチング
        for relation_type, keywords in self.relation_patterns.items():
            for keyword in keywords:
                if keyword in between_text:
                    return relation_type
        
        # エンティティタイプベースの関係推論
        if ent1.label == 'PERSON' and ent2.label == 'INSURANCE_PRODUCT':
            return 'CONTRACTED'
        elif ent1.label == 'ORGANIZATION' and ent2.label == 'DOCUMENT':
            return 'ISSUED_BY'
        elif ent1.label == 'INSURANCE_PRODUCT' and ent2.label == 'COVERAGE':
            return 'COVERS'
        elif ent1.label in ['PERSON', 'ORGANIZATION'] and ent2.label == 'MONEY':
            return 'PAYS'
        
        # デフォルトの関係
        if ent1.label != ent2.label:
            return 'RELATED_TO'
        
        return None

# Neo4j知識グラフ管理クラスは変更なし
class Neo4jKnowledgeGraph:
    """Neo4j知識グラフ管理"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def _create_constraints(self):
        """制約とインデックスの作成"""
        with self.driver.session() as session:
            try:
                # 実体の一意性制約
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                
                # インデックス作成
                session.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)")
                
                logger.info("Neo4j制約とインデックスの作成完了")
            except Exception as e:
                logger.warning(f"Neo4j制約作成エラー（無視可能）: {e}")

    def add_entity(self, entity: Entity, document_id: str = None):
        """実体をグラフに追加"""
        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {name: $name, type: $type})
            SET e.text = $text, 
                e.confidence = $confidence,
                e.document_id = $document_id,
                e.updated_at = datetime()
            RETURN e
            """
            session.run(query, 
                       name=entity.text,
                       type=entity.label,
                       text=entity.text,
                       confidence=entity.confidence,
                       document_id=document_id)

    def add_relation(self, relation: Relation, document_id: str = None):
        """関係をグラフに追加"""
        with self.driver.session() as session:
            # 動的な関係タイプのためのクエリ構築
            query = f"""
            MATCH (source:Entity {{name: $source}})
            MATCH (target:Entity {{name: $target}})
            MERGE (source)-[r:`{relation.relation_type}`]->(target)
            SET r.confidence = $confidence,
                r.document_id = $document_id,
                r.updated_at = datetime()
            RETURN r
            """
            try:
                session.run(query,
                           source=relation.source,
                           target=relation.target,
                           confidence=relation.confidence,
                           document_id=document_id)
            except Exception as e:
                logger.error(f"関係追加エラー: {e}")

    def query_graph(self, query: str) -> List[Dict]:
        """グラフクエリの実行"""
        with self.driver.session() as session:
            try:
                result = session.run(query)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"クエリ実行エラー: {e}")
                return []

    def get_entity_relationships(self, entity_name: str) -> List[Dict]:
        """特定実体の関係を取得"""
        query = """
        MATCH (e:Entity {name: $name})-[r]->(target)
        RETURN e.name as source, type(r) as relationship, target.name as target, r.confidence as confidence
        UNION
        MATCH (source)-[r]->(e:Entity {name: $name})
        RETURN source.name as source, type(r) as relationship, e.name as target, r.confidence as confidence
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, name=entity_name)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"関係取得エラー: {e}")
                return []

    def close(self):
        """接続を閉じる"""
        self.driver.close()

class InsuranceKnowledgeGraphBuilder:
    """保険知識グラフ構築メインクラス"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 tokenizer: str = 'auto'):
        self.nlp_processor = JapaneseNLPProcessor(prefer_tokenizer=tokenizer)
        self.knowledge_graph = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

    def process_document(self, text: str, document_id: str = None) -> Dict:
        """文書を処理して知識グラフを構築"""
        logger.info(f"文書処理開始: {document_id}")
        
        try:
            # 実体抽出
            entities = self.nlp_processor.extract_entities(text)
            logger.info(f"抽出された実体数: {len(entities)}")
            
            # 関係抽出
            relations = self.nlp_processor.extract_relations(text, entities)
            logger.info(f"抽出された関係数: {len(relations)}")
            
            # Neo4jに追加
            for entity in entities:
                self.knowledge_graph.add_entity(entity, document_id)
            
            for relation in relations:
                self.knowledge_graph.add_relation(relation, document_id)
            
            return {
                'document_id': document_id,
                'tokenizer_used': type(self.nlp_processor.tokenizer).__name__,
                'entities': [{'text': e.text, 'label': e.label, 'confidence': e.confidence} for e in entities],
                'relations': [{'source': r.source, 'target': r.target, 'type': r.relation_type, 'confidence': r.confidence} for r in relations]
            }
        
        except Exception as e:
            logger.error(f"文書処理エラー: {e}")
            return {
                'document_id': document_id,
                'error': str(e),
                'entities': [],
                'relations': []
            }

    def analyze_knowledge_graph(self) -> Dict:
        """知識グラフの分析"""
        try:
            # エンティティ統計
            entity_stats = self.knowledge_graph.query_graph("""
            MATCH (e:Entity)
            RETURN e.type as type, count(e) as count
            ORDER BY count DESC
            """)
            
            # 関係統計
            relation_stats = self.knowledge_graph.query_graph("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """)
            
            # 最も接続の多いエンティティ
            top_entities = self.knowledge_graph.query_graph("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as connections
            RETURN e.name as entity, e.type as type, connections
            ORDER BY connections DESC
            LIMIT 10
            """)
            
            return {
                'entity_statistics': entity_stats,
                'relation_statistics': relation_stats,
                'top_connected_entities': top_entities
            }
        
        except Exception as e:
            logger.error(f"グラフ分析エラー: {e}")
            return {}

    def close(self):
        """リソースを解放"""
        self.knowledge_graph.close()

# 使用例とテスト
def main():
    """メイン関数"""
    # Neo4j接続設定
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # 実際のパスワードに変更してください
    
    # システム初期化（分词器を指定可能）
    kg_builder = InsuranceKnowledgeGraphBuilder(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
        tokenizer='auto'  # 'sudachi', 'janome', 'spacy', 'regex', 'auto'
    )
    
    # サンプル日本語保険文書
    sample_text = """
    契約者の田中太郎様は、令和6年4月1日にABC生命保険株式会社の終身保険に加入されました。
    被保険者は田中太郎様ご本人で、受益者は配偶者の田中花子様となっています。
    年間保険料は120万円、死亡保障額は3000万円です。
    保険証券番号はL-2024-001234で、契約満期日は設定されていません。
    この契約には医療特約も付帯しており、入院給付金は日額1万円となっています。
    代理店の山田保険事務所が契約手続きを担当いたしました。
    """
    
    try:
        # 文書処理
        result = kg_builder.process_document(sample_text, "insurance_doc_001")
        print("=== 処理結果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 知識グラフ分析
        analysis = kg_builder.analyze_knowledge_graph()
        print("\n=== 知識グラフ分析 ===")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        
        # 特定エンティティの関係取得
        if result['entities']:
            first_entity = result['entities'][0]['text']
            relationships = kg_builder.knowledge_graph.get_entity_relationships(first_entity)
            print(f"\n=== {first_entity} の関係 ===")
            for rel in relationships:
                print(f"{rel['source']} --[{rel['relationship']}]--> {rel['target']}")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
    
    finally:
        kg_builder.close()

if __name__ == "__main__":
    main()