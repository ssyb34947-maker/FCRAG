# **Financial Cryptocurrency Retrieval-Augmented Generation**

这是一个为金融领域行业智能体设计的RAG系统。提供完整的端到端的知识管理功能，和可视化操作界面。利用大语言模型进行知识处理，极大提高了多源知识表示能力。

## 功能特点

- 多种文档格式支持（TXT、PDF、DOCX、HTML）
- 灵活的文本分块策略（滑动窗口、句子、段落、混合）
- 多种去重机制（MD5、SimHash、Embedding 相似度）
- Milvus 向量数据库
- 可配置的检索和重排策略
- 大语言模型知识表示
- 时间段检索支持
- 完整的 CLI 接口

## 项目结构

```
rag_system/
├── config.yaml                 # 配置文件
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── loaders/
│   └── loader_manager.py       # 文档加载器管理器
├── chunking/
│   └── splitter.py             # 文本分割器
├── embeddings/
│   └── bairen_embedder.py      # 阿里云百炼嵌入器
├── storage/
│   └── milvus_client.py        # Milvus 客户端
├── retrieval/
│   ├── searcher.py             # 检索器
│   └── reranker.py             # 重排器
├── prompts/
│   └── knowledge_integration.yaml  # 知识整合提示词模板
├── utils/
│   ├── logger.py               # 日志工具
│   ├── tokenizer.py            # 分词器
│   ├── dedup.py                # 去重工具
│   └── llm_processor.py        # 大语言模型处理器
├── backend/
│	└── main.py
├── frontend/
│	└── index.html
├── test/
└── README.md                   # 项目说明文档
```

## 环境依赖

- Python 3.8+
- Milvus 2.0+
- Python依赖见requirement.txt

## 安装步骤

1. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装并启动 Milvus：
   参考 [Milvus 官方文档](https://milvus.io/docs/install_standalone-docker.md) 安装并启动 Milvus。

4. 配置参数：
   编辑 `config.yaml` 文件，设置 Milvus 连接参数、阿里云百炼 API Key 或其他大语言模型 API 配置。

## 配置说明

在 `config.yaml` 文件中配置以下参数：

### Milvus 配置
```yaml
milvus:
  host: YOURHOST           # Milvus 服务器地址
  port: YOURPORT              # Milvus 服务器端口
  user: ""                    # 用户名（可选）
  password: ""                # 密码（可选）
  collection_name: "YOURCOLLECTIONNAME"  # 集合名称
  dim: 128                    # 向量维度
  index_params:               # 索引参数
    metric_type: "IP"         # 度量类型（内积）
    index_type: "IVF_FLAT"    # 索引类型
    params:
      nlist: 128              # 聚类数量
```

### Embedding 配置
```yaml
embedding:
  api_key: "YOUREMBEDDINGAPIKEY"  # 阿里云百炼 API Key
  model_name: "qwen2.5-vl-embedding"       # 模型名称
  batch_size: 1                   # 批处理大小
  timeout: 30                      # 超时时间（秒）
  max_retries: 3                   # 最大重试次数
```

### Chunking 配置
```yaml
chunking:
  strategy: "sliding_token"        # 分块策略（sliding_token/sentence/paragraph/hybrid）
  chunk_size: 512                  # 块大小（token 数量）
  chunk_overlap: 64                # 重叠大小（token 数量）
```

### Dedup 配置
```yaml
dedup:
  strategy: "simhash"              # 去重策略（md5/simhash/embedding）
  md5_threshold: 1.0               # MD5 阈值
  simhash_threshold: 3             # SimHash 阈值
  embedding_threshold: 0.95        # Embedding 阈值
```

### Retrieval 配置
```yaml
retrieval:
  top_k: 10                        # 检索返回结果数量
  domain_filter: true              # 是否启用域过滤
```

### Reranker 配置
```yaml
reranker:
  mode: "embedding_bm25_mixed"     # 重排模式（cross_encoder/embedding_bm25_mixed）
  weights:                         # 权重配置
    ann: 0.7                       # 近似最近邻得分权重
    bm25: 0.3                      # BM25 得分权重
```

### LLM 配置
```yaml
llm:
  api_key: "your_llm_api_key"      # 大语言模型 API Key
  base_url: "https://api.deepseek.com"  # 兼容 OpenAI 范式的 API 地址
  model_name: "deepseek-chat"        # 模型名称
  timeout: 30                      # 超时时间（秒）
  max_retries: 3                   # 最大重试次数
```

### Logging 配置
```yaml
logging:
  level: "INFO"                    # 日志级别
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志格式
  file: "rag_system.log"           # 日志文件路径
```

## 使用方法

### 1. 添加文档到知识库

添加单个文件：
```bash
python main.py ingest --path /path/to/document.pdf --domain 经济学
```

添加整个目录：
```bash
python main.py ingest --path /path/to/documents --domain 经济学
```

### 2. 查询知识库

基本查询：
```bash
python main.py query --q "什么是有效市场假说？" --domain 经济学
```

跨域查询：
```bash
python main.py query --q "经济学的基本原理是什么？"
```

### 3. 在代码中使用 RAGEngine

```python
from main import RAGEngine

# 初始化 RAG 引擎
rag = RAGEngine()

# 添加文档
rag.ingest("/path/to/documents", "经济学")

# 查询
results = rag.query("什么是有效市场假说？", "经济学")
for result in results:
    print(f"得分: {result['final_score']}")
    print(f"内容: {result['content'][:100]}...")在代码中使用 RAGEngine
```

4.通过可视化界面操作

- 启动后端

```bash
cd backend
uvicorn backend.main:app --reload
```

- 启动前端，打开frontend/index.html

![FCRAG/img/img.png at main · ssyb34947-maker/FCRAG](https://github.com/ssyb34947-maker/FCRAG/blob/main/img/img.png)

## 功能说明

### 大语言模型知识整合

系统现在会在文本分块后自动调用大语言模型对知识内容进行整合和结构化处理。这个过程包括：

1. 将原始文本传递给大语言模型
2. 使用预定义的提示词模板引导模型输出标准化的 JSON 结构
3. 提取结构化信息，包括内容、类型、时间戳和来源

### 时间段检索

数据库现在包含了时间戳字段，可以支持基于时间段的检索功能，方便用户按时间范围查找相关信息。

### 工程化提示词管理

所有大语言模型的提示词都被集中管理在 [prompts/knowledge_integration.yaml](file:///d:/%E6%A1%8C%E9%9D%A2/work/%E8%8A%B1%E6%97%97%E6%9D%AF/RAG/prompts/knowledge_integration.yaml) 文件中，便于维护和更新。

## 单元测试

项目包含完整的单元测试，可以在没有 Milvus 环境的情况下测试各个组件：

### 运行所有测试：

```bash
python -m pytest test/ -v
```

或者使用测试运行器：

```bash
python test/run_all_tests.py
```

### 运行特定模块测试：

```bash
# 测试数据加载器
python -m pytest test/test_loader.py -v

# 测试分块功能
python -m pytest test/test_chunking.py -v

# 测试嵌入功能
python -m pytest test/test_embedding.py -v

# 测试去重功能
python -m pytest test/test_dedup.py -v

# 测试检索功能
python -m pytest test/test_searcher.py -v

# 测试重排功能
python -m pytest test/test_reranker.py -v
```

## 开发指南

### 添加新的文档加载器

在 `loaders/loader_manager.py` 中的 `loader_mapping` 字典中添加新的文件扩展名和对应的加载器。

### 添加新的分块策略

在 `chunking/splitter.py` 中添加新的分块方法，并在 `split` 方法中添加相应的条件分支。

### 添加新的去重策略

在 `utils/dedup.py` 中添加新的去重方法，并在 `is_duplicate` 方法中添加相应的条件分支。

### 更新提示词模板

编辑 `prompts/knowledge_integration.yaml` 文件来更新大语言模型的提示词模板。

## 注意事项

1. 确保 Milvus 服务正在运行
2. 设置正确的阿里云百炼 API Key 或其他大语言模型 API 配置
3. 根据使用的 Embedding 模型调整向量维度
4. 根据实际需求调整分块大小和重叠度
5. 定期备份 Milvus 数据

## 许可证

本项目采用 MIT 许可证，详情请见 LICENSE 文件。