# LocalRAG

> 🚀 **Your AI's Second Brain** - A privacy-first local knowledge base powered by BAAI models and MCP protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

**LocalRAG** 是一个开箱即用的本地知识库检索系统，专为 AI 助手（如 Claude Desktop）设计。通过 MCP 协议连接，让您的 AI 助手能够：

- 📚 **智能管理文档**：支持 PDF、TXT、Markdown 多格式导入
- 🔍 **精准检索答案**：基于 BAAI bge-m3 向量模型 + 可选重排模型
- 🔒 **完全本地化**：所有数据存储在本地，隐私零泄露
- ⚡ **高效异步处理**：后台队列处理大批量文档
- 🎯 **多项目隔离**：支持创建多个独立知识库

一个基于 Model Context Protocol (MCP) 的本地知识库 RAG (Retrieval-Augmented Generation) 服务器，使用 BAAI 的 bge-m3 向量模型和重排模型实现高效的文档检索与问答。所有数据本地存储，保护您的隐私安全。

## ✨ 核心特性

- 🔍 **智能向量化检索**: 使用 BAAI/bge-m3 高质量中文向量模型
- 🎯 **精准重排序**: 可选的 BAAI/bge-reranker-m3 重排模型,大幅提升检索精度
- 📁 **多格式支持**: 支持 PDF、TXT、Markdown 文件自动解析
- ⚡ **异步处理**: 后台队列处理大量文档入库,不阻塞主服务
- 🗂️ **多项目隔离**: 支持创建多个独立的知识库集合 (Collection)
- 💾 **本地持久化**: 所有数据存储在本地 ChromaDB,数据安全可控
- 🚀 **GPU 加速**: 自动检测 CUDA 环境,优先使用 GPU 加速推理

## 📋 系统要求

- Python 3.10+
- 支持 CUDA 的 GPU (可选,推荐)
- 至少 8GB RAM (CPU 模式) / 4GB VRAM (GPU 模式)

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install mcp chromadb sentence-transformers pypdf FlagEmbedding
```

**GPU 用户额外安装**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 配置路径 (可选)

默认配置将数据存储在 `D:\baai_rag` 目录。如需修改,编辑 `baai_rag_server.py` 中的以下常量:

```python
BASE_DIR = r"D:\baai_rag"  # 修改为你的目标路径
DB_PATH = os.path.join(BASE_DIR, "chroma_db_data")  # 数据库存储路径
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_models")  # 模型缓存路径
```

### 3. 运行服务器

```bash
python baai_rag_server.py
```

首次运行会自动从 Hugging Face 下载模型 (约 2GB),请耐心等待。

### 4. 在 Claude Desktop 中配置

编辑 Claude Desktop 配置文件:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`  
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

添加以下配置:

```json
{
  "mcpServers": {
    "local-knowledge": {
      "command": "python",
      "args": ["D:/baai_rag/baai_rag_server.py"]
    }
  }
}
```

重启 Claude Desktop 后即可使用。

## 🛠️ 工具说明

此 MCP 服务器提供以下 3 个工具:

### 1. `add_file_to_knowledge` - 文档入库

将本地文档向量化并存入知识库。

**参数**:
- `file_path` (必需): 文件的完整路径,如 `"D:/项目/文档/论文1.pdf"`
- `collection_name` (必需): 项目数据库名称,仅支持小写字母、数字、下划线,最少 3 个字符,如 `"nsfc2026"`

**特点**:
- 异步处理,立即返回状态
- 自动提取文本内容 (PDF/TXT/MD)
- 智能滑动窗口分块 (chunk_size=600, overlap=100)
- 支持批量文件导入

**示例**:
```
用户: 请把 D:/论文/注意力机制.pdf 和 D:/论文/transformer.md 都导入到 my_research 数据库

AI: [调用工具] add_file_to_knowledge("D:/论文/注意力机制.pdf", "my_research")
    [调用工具] add_file_to_knowledge("D:/论文/transformer.md", "my_research")
    
    🚀 注意力机制.pdf 已加入排队
    🚀 transformer.md 已加入排队
```

### 2. `check_knowledge_status` - 队列状态查询

检查后台入库队列的处理进度。

**参数**: 无

**返回**: 当前队列中待处理文件数量

**示例**:
```
用户: 文件处理完了吗?

AI: [调用工具] check_knowledge_status()
    
    ⏳ 后台模型正在奋力运算中,当前队列还有 5 个文件正在排队等待向量化!
```

### 3. `search_knowledge` - 知识库检索

在指定知识库中搜索相关内容。

**参数**:
- `query` (必需): 检索问题,如 `"项目进度的风险有哪些?"`
- `collection_name` (必需): 要搜索的项目数据库名称

**特点**:
- 第一阶段: 向量检索召回 Top 15 (有 Reranker) 或 Top 3 (无 Reranker)
- 第二阶段 (可选): BGE-Reranker 精排,返回最相关的 Top 3
- 返回文档片段及来源路径

**示例**:
```
用户: 在 my_research 库中搜索"注意力机制的计算复杂度如何优化?"

AI: [调用工具] search_knowledge("注意力机制的计算复杂度如何优化?", "my_research")
    
    🎯 Rerank 精排名单 (查找到 15 条,已截取最相关的 Top 3):
    
    --- 来源: D:/论文/注意力机制.pdf ---
    注意力机制的时间复杂度为 O(n²),其中 n 为序列长度...
    
    --- 来源: D:/论文/transformer.md ---
    为了降低计算复杂度,可以采用稀疏注意力机制...
```

## 📊 工作流程图

```
用户请求
   ↓
MCP Server 接收
   ↓
┌─────────────┬──────────────┐
│   入库操作   │    检索操作   │
└─────────────┴──────────────┘
      ↓               ↓
  异步队列         向量检索 (Top 15)
      ↓               ↓
  文本提取         Rerank 精排
      ↓               ↓
  向量化存储      返回 Top 3 结果
      ↓
  ChromaDB 持久化
```

## 🔧 高级配置

### 调整分块参数

修改 `chunk_text` 函数:

```python
def chunk_text(text: str, chunk_size=600, overlap=100) -> list:
    # chunk_size: 每个文本块的字符数
    # overlap: 相邻块之间的重叠字符数
```

### 调整检索数量

修改 `search_knowledge` 函数中的参数:

```python
recall_num = 15  # 向量检索召回数量 (有 Reranker 时)
# 或
recall_num = 3   # 无 Reranker 时直接返回数量
```

### 禁用 GPU

如果 GPU 内存不足,可以强制使用 CPU:

```python
best_device = "cpu"  # 在代码第 32 行强制设置为 CPU
```

## 📝 使用场景

### 1. 学术研究管理
```bash
# 创建多个项目库
add_file_to_knowledge("论文1.pdf", "transformer_research")
add_file_to_knowledge("论文2.pdf", "transformer_research")
add_file_to_knowledge("综述.md", "survey_2024")

# 精准检索
search_knowledge("Transformer 的位置编码方式有哪些?", "transformer_research")
```

### 2. 企业知识库
```bash
# 导入公司文档
add_file_to_knowledge("产品手册.pdf", "company_kb")
add_file_to_knowledge("技术文档.md", "company_kb")

# 快速问答
search_knowledge("如何配置 API 密钥?", "company_kb")
```

### 3. 个人笔记系统
```bash
# 管理个人知识
add_file_to_knowledge("读书笔记.md", "my_notes")
add_file_to_knowledge("会议纪要.pdf", "work_notes")

# 知识回顾
search_knowledge("上次项目会议的决策点", "work_notes")
```

## ⚠️ 注意事项

1. **首次运行**: 会自动下载约 2GB 的模型文件,请确保网络畅通
2. **路径格式**: 建议使用正斜杠 `/` 或双反斜杠 `\\`,避免路径转义问题
3. **集合命名**: 仅支持小写字母、数字、下划线,长度至少 3 字符
4. **内存占用**: 
   - CPU 模式: 约 4GB RAM
   - GPU 模式: 约 2GB VRAM
5. **并发处理**: 队列采用单线程处理,避免资源竞争

## 🐛 常见问题

### Q: 提示找不到文件,但路径明明正确?
A: 可能是中文引号或特殊字符被转义。脚本已内置 `resolve_fuzzy_path` 函数自动修复,如仍有问题请使用英文路径。

### Q: 如何查看已创建的知识库列表?
A: 当前版本暂未提供列表工具,可以通过 ChromaDB 的 API 查看:

```python
import chromadb
client = chromadb.PersistentClient(path="D:/baai_rag/chroma_db_data")
print(client.list_collections())
```

### Q: Reranker 加载失败?
A: Reranker 是可选组件,不影响基础功能。如需启用:
```bash
pip install -U FlagEmbedding
```

### Q: 如何删除错误导入的文件?
A: 当前版本暂未提供删除工具,可以通过 ChromaDB API 删除整个集合:

```python
client.delete_collection("错误的集合名")
```

## 📚 技术栈

- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) - AI 工具协议
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 中文向量模型
- [BAAI/bge-reranker-m3](https://huggingface.co/BAAI/bge-reranker-m3) - 重排模型
- [Sentence-Transformers](https://www.sbert.net/) - 向量化框架
- [PyPDF](https://pypdf.readthedocs.io/) - PDF 解析

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📧 联系方式

如有问题或建议,请提交 GitHub Issue。

---

**Star ⭐ 本项目以获取最新更新!**