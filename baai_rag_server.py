import os
import sys
import threading
import queue
import re
import time
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

# --- 1. 基础配置 ---
import logging
# 禁用所有默认的 logging，避免污染 MCP 的 stdout
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(message)s')
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

BASE_DIR = r"D:\baai_rag"
DB_PATH = os.path.join(BASE_DIR, "chroma_db_data")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_models")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# 强制 Python 不要缓存 stdout 输出，避免 MCP 通信挂起
os.environ["PYTHONUNBUFFERED"] = "1"

# 初始化 MCP
mcp = FastMCP("Local Knowledge (BAAI)")

# --- 2. 硬件加速及模型懒加载 ---
# 全局变量占位
best_device = None
ef = None
reranker = None
client = None
_initialized = False

def detect_device() -> str:
    """尝试使用 CUDA，失败则使用 CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 已启用 CUDA GPU 加速！", file=sys.stderr)
            return "cuda"
    except ImportError:
        pass
    print("未能使用 CUDA，已回退至 CPU 运行。", file=sys.stderr)
    return "cpu"

def lazy_init():
    global best_device, ef, reranker, client, _initialized
    if _initialized:
        return
        
    import sys
    best_device = detect_device()
    print("正在加载 BAAI 向量模型 (BAAI/bge-m3)...", file=sys.stderr)

    # 强制将模型加载过程中的所有 print 重定向到 stderr
    _original_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        # 初始化 BAAI/bge-m3 中文优化模型
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=best_device 
        )

        # 尝试加载 BGE 精排模型
        try:
            from FlagEmbedding import FlagReranker
            print("正在加载 BAAI 重排模型 (BAAI/bge-reranker-m3)...", file=sys.stderr)
            use_half_precision = (best_device == "cuda")
            reranker = FlagReranker('BAAI/bge-reranker-m3', use_fp16=use_half_precision)
            print("✅ 重排模型加载成功，搜索精排已启用！", file=sys.stderr)
        except ImportError:
            reranker = None
            print("⚠️ 提示: 未检测到 FlagEmbedding 库。", file=sys.stderr)
    finally:
        sys.stdout = _original_stdout

    # 获取数据库连接
    client = chromadb.PersistentClient(path=DB_PATH)
    print("✅ 数据库加载成功，服务启动完毕！", file=sys.stderr)
    _initialized = True



# --- 2. 辅助函数集 (代码瘦身) ---

def extract_text(file_path: str) -> str:
    """自动根据后缀提取文本内容"""
    ext = file_path.lower()
    text = ""
    if ext.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif ext.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError("不支持该文件类型")
    return text

def chunk_text(text: str, chunk_size=600, overlap=100) -> list:
    """文本定长滑动窗口分块"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def resolve_fuzzy_path(file_path: str) -> str:
    """修复大模型可能导致的中文引号转义问题"""
    if os.path.exists(file_path):
        return file_path
        
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    
    if os.path.exists(dir_name):
        def strip_quotes(s):
            return s.replace('"', '').replace('“', '').replace('”', '').replace("'", "")
        target_stripped = strip_quotes(base_name)
        for f in os.listdir(dir_name):
            if strip_quotes(f) == target_stripped:
                return os.path.join(dir_name, f)
    return file_path

def sanitize_collection_name(name: str) -> str:
    """净化集合名称，符合 ChromaDB 规范"""
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_').lower()
    if not safe_name or len(safe_name) < 3:
        raise ValueError(f"提供的集合名称 '{name}' 不合法。至少需包含3个英文字母或数字。")
    return safe_name


# --- 3. 后台入库工作线程 ---

task_queue = queue.Queue()

def background_worker():
    while True:
        task = task_queue.get()
        if task is None: break
        
        fp = task['file_path']
        cname = task['collection_name']
        
        try:
            print(f"[排队执行中] 开始处理文档: {os.path.basename(fp)}", file=sys.stderr)
            
            # 1. 提取与分块
            text = extract_text(fp)
            if not text.strip():
                print(f"[任务跳过] {fp} 内容为空。", file=sys.stderr)
                continue
                
            chunks = chunk_text(text)
            
            # 2. 准备向量数据标识
            file_name = os.path.basename(fp)
            timestamp = int(time.time())
            ids = [f"{file_name}_{timestamp}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": fp, "chunk_index": i} for i in range(len(chunks))]

            # 3. 存入专属数据库
            collection = client.get_or_create_collection(name=cname, embedding_function=ef)
            collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
            print(f"[入库成功({cname})] ✅ 《{file_name}》处理完毕。队列剩余: {task_queue.qsize()}个", file=sys.stderr)

        except Exception as e:
            print(f"[入库失败({cname})] ❌ 处理《{os.path.basename(fp)}》失败: {str(e)}", file=sys.stderr)
        finally:
            task_queue.task_done()

# 启动唯一的后台处理线程
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()


# --- 4. MCP 工具定义 ---

@mcp.tool()
def add_file_to_knowledge(file_path: str, collection_name: str) -> str:
    """
    [入库] 读取本地 PDF/TXT/MD 文件，使用 BAAI 模型向量化并存入 D 盘数据库。
    【重要指令给AI大模型】：
    1. 此工具在后台异步运行！你只需要传入路径即可，马上就可以继续处理下一个文件或向用户报告进度。
    2. collection_name 参数是强制的。它是你为这个特定项目建立的数据库名称（必须是小写字母、数字或下划线组成，最短3个字符）。例如你可以传入 "nsfc2026"。
    Args:
        file_path: 文件的完整路径，例如 "D:/项目/NSFC2026/耐心资本/论文1.md"
        collection_name: 强制要求！你想要放入的"项目数据库"名称（只能是小写字母/数字/下划线）。例如 "nsfc2026"
    """
    lazy_init()
    file_path = resolve_fuzzy_path(file_path)
    if not os.path.exists(file_path):
        return f"错误：找不到文件 {file_path}。请检查是否因为中文特殊符号(比如引号)被错误转义。"

    try:
        safe_name = sanitize_collection_name(collection_name)
    except ValueError as e:
        return f"错误：{str(e)}"

    task_queue.put({'file_path': file_path, 'collection_name': safe_name})
    return f"🚀 {os.path.basename(file_path)} 已加入排队（库: {safe_name}）。这可能是由于你有大批量文件，可用 check_knowledge_status 查看剩余数量。"

@mcp.tool()
def check_knowledge_status() -> str:
    """
    [查询状态] 检查当前后台 RAG 知识库入库队列中还有多少个文件正在排队等待处理。
    【重要指令给AI大模型】：由于文件入库是异步执行的，你可以定期调用此工具以便告知用户当前的后台进度。
    """
    qsize = task_queue.qsize()
    if qsize == 0:
        return "✅ 当前后台向量库入库队列已清空，所有文件已处理完毕！"
    else:
        return f"⏳ 后台模型正在奋力运算中，当前队列还有 {qsize} 个文件正在排队等待向量化！"

@mcp.tool()
def search_knowledge(query: str, collection_name: str) -> str:
    """
    [查询] 在 D 盘本地知识库中搜索相关内容。
    【重要指令给AI大模型】：你必须显式指定在哪一个独立数据库（collection_name）中进行搜索。
    Args:
        query: 你的问题，例如 "项目进度的风险有哪些？"
        collection_name: 强制要求！你想要搜索的具体"项目数据库"名称。例如 "nsfc2026"。
    """
    lazy_init()
    try:
        safe_name = sanitize_collection_name(collection_name)
        collection = client.get_collection(name=safe_name, embedding_function=ef)
    except ValueError as e:
        return f"错误：{str(e)}"
    except Exception:
        return f"❌ 找不到名为 '{safe_name}' 的项目知识库。该知识库可能还未被创建，或者您传错名字了。"
            
    try:
        # 第一阶段：向量初排召回 (根据是否加载重排模型决定召回数量)
        recall_num = 15 if reranker else 3
        results = collection.query(
            query_texts=[query],
            n_results=recall_num
        )
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        if not docs:
            return "知识库里没有找到相关内容。"
            
        # 第二阶段：BGE-Reranker 精排
        if reranker and len(docs) > 1:
            # 构造 [query, doc] 对
            pairs = [[query, doc] for doc in docs]
            # 计算相似度得分
            scores = reranker.compute_score(pairs)
            
            # 根据得分倒序排列组合
            scored_docs = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
            
            # 取出得分最高的 Top 3
            final_docs = [item[1] for item in scored_docs[:3]]
            final_metas = [item[2] for item in scored_docs[:3]]
            response = f"🎯 Rerank 精排名单 (查找到 {len(docs)} 条，已截取最相关的 Top 3):\n"
        else:
            final_docs = docs[:3]
            final_metas = metas[:3]
            response = f"🔍 向量检索名单 (Top {len(final_docs)}):\n"
            
        # 组装返回结果
        for i, doc in enumerate(final_docs):
            src = final_metas[i].get('source', '未知文件')
            response += f"\n--- 来源: {src} ---\n{doc.strip()}\n"
            
        return response
    except Exception as e:
        return f"查询出错: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')