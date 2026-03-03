import os
import sys
import logging
import threading
import queue
import re
import time
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

# --- 配置：日志 / 环境变量 ---
# 所有 logging 输出强制走 stderr，避免污染 MCP 的 stdout JSON-RPC 通道
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(message)s')
for lib in ["sentence_transformers", "httpx", "httpcore", "huggingface_hub"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

BASE_DIR = r"D:\baai_rag"
DB_PATH  = os.path.join(BASE_DIR, "chroma_db_data")
os.environ["HF_HOME"]                 = os.path.join(BASE_DIR, "hf_models")
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# --- MCP 服务初始化 ---
mcp = FastMCP("Local Knowledge (BAAI)")

# --- 模型状态（懒加载） ---
best_device  = None
ef           = None   # 向量化模型
reranker     = None   # 精排模型（可选）
db_client    = None   # ChromaDB 客户端
_initialized = False
_init_lock   = threading.Lock()


def detect_device() -> str:
    """自动检测 CUDA 或回退 CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，已启用 CUDA 加速！", file=sys.stderr)
            return "cuda"
    except ImportError:
        pass
    print("未检测到 CUDA，使用 CPU 运行。", file=sys.stderr)
    return "cpu"


def lazy_init():
    """加载向量模型、精排模型和数据库（首次调用时执行）"""
    global best_device, ef, reranker, db_client, _initialized
    if _initialized:
        return

    best_device = detect_device()
    print("正在加载 BAAI/bge-m3 向量模型...", file=sys.stderr)

    # 临时将 stdout 重定向到 stderr，防止底层库输出污染 MCP 通道
    _orig_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=best_device
        )
        print("✅ 向量模型加载完毕。", file=sys.stderr)

        try:
            from FlagEmbedding import FlagReranker
            print("正在加载 BAAI/bge-reranker-m3 精排模型...", file=sys.stderr)
            reranker = FlagReranker("BAAI/bge-reranker-m3", use_fp16=(best_device == "cuda"))
            print("✅ 精排模型加载完毕，Rerank 已启用。", file=sys.stderr)
        except ImportError:
            reranker = None
            print("⚠️  未检测到 FlagEmbedding，精排功能已跳过。", file=sys.stderr)
    finally:
        sys.stdout = _orig_stdout

    db_client    = chromadb.PersistentClient(path=DB_PATH)
    _initialized = True
    print("✅ 数据库就绪，MCP 服务已完全启动！", file=sys.stderr)


def background_init():
    """服务启动后在后台线程预热模型，避免首次工具调用阻塞"""
    with _init_lock:
        try:
            lazy_init()
        except Exception as e:
            import traceback
            print(f"❌ 后台预热失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

threading.Thread(target=background_init, daemon=True).start()


# --- 辅助函数 ---

def extract_text(file_path: str) -> str:
    """根据文件类型提取纯文本"""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file_path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"不支持的文件类型: {os.path.splitext(file_path)[1]}")


def chunk_text(text: str, chunk_size=600, overlap=100) -> list:
    """滑动窗口分块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def resolve_fuzzy_path(file_path: str) -> str:
    """修复 AI 可能引入的中文引号或转义问题"""
    if os.path.exists(file_path):
        return file_path
    dir_name, base_name = os.path.dirname(file_path), os.path.basename(file_path)
    if os.path.exists(dir_name):
        strip = lambda s: s.replace('"','').replace('“','').replace('”','').replace("'",'').replace('‘','').replace('’','')
        for f in os.listdir(dir_name):
            if strip(f) == strip(base_name):
                return os.path.join(dir_name, f)
    return file_path


def sanitize_collection_name(name: str) -> str:
    """将任意字符串净化为合法的 ChromaDB 集合名"""
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_').lower()
    if len(safe) < 3:
        raise ValueError(f"集合名称 '{name}' 不合法，至少需要 3 个字母/数字。")
    return safe


# --- 后台入库队列 ---

task_queue = queue.Queue()

def background_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        fp, cname = task["file_path"], task["collection_name"]
        try:
            # 若模型还在预热，等待锁后继续
            if not _initialized:
                with _init_lock:
                    lazy_init()
            print(f"[入库] 开始处理: {os.path.basename(fp)}", file=sys.stderr)
            text = extract_text(fp)
            if not text.strip():
                print(f"[跳过] {fp} 内容为空", file=sys.stderr)
                continue
            chunks    = chunk_text(text)
            fname     = os.path.basename(fp)
            ts        = int(time.time())
            ids       = [f"{fname}_{ts}_{i}" for i in range(len(chunks))]
            metas     = [{"source": fp, "chunk_index": i} for i in range(len(chunks))]
            col       = db_client.get_or_create_collection(name=cname, embedding_function=ef)
            col.upsert(documents=chunks, ids=ids, metadatas=metas)
            print(f"[入库成功] ✅ 《{fname}》→ {cname}，队列剩余 {task_queue.qsize()} 个", file=sys.stderr)
        except Exception as e:
            print(f"[入库失败] ❌ {os.path.basename(fp)}: {e}", file=sys.stderr)
        finally:
            task_queue.task_done()

threading.Thread(target=background_worker, daemon=True).start()


# --- MCP 工具定义 ---

@mcp.tool()
def add_file_to_knowledge(file_path: str, collection_name: str) -> str:
    """
    [入库] 读取本地 PDF/TXT/MD 文件，向量化后存入本地知识库。
    此工具立即返回，文件在后台异步处理。可用 check_knowledge_status 查看进度。
    Args:
        file_path: 文件完整路径，例如 "D:/项目/论文.pdf"
        collection_name: 知识库名称（小写字母/数字/下划线，最少3字符），例如 "nsfc2026"
    """
    file_path = resolve_fuzzy_path(file_path)
    if not os.path.exists(file_path):
        return f"❌ 找不到文件: {file_path}"
    try:
        safe_name = sanitize_collection_name(collection_name)
    except ValueError as e:
        return f"❌ {e}"
    task_queue.put({"file_path": file_path, "collection_name": safe_name})
    return f"🚀 《{os.path.basename(file_path)}》已加入队列（库: {safe_name}）。可用 check_knowledge_status 查看进度。"


@mcp.tool()
def check_knowledge_status() -> str:
    """[状态] 查看后台入库队列中还有多少文件待处理。"""
    n = task_queue.qsize()
    if not _initialized:
        return f"⏳ 模型正在后台预热（约需 20-30 秒），队列中有 {n} 个文件等待处理。"
    return f"✅ 队列已清空，所有文件处理完毕！" if n == 0 else f"⏳ 队列中还有 {n} 个文件正在处理。"


@mcp.tool()
def search_knowledge(query: str, collection_name: str) -> str:
    """
    [检索] 在本地知识库中语义搜索相关内容。
    Args:
        query: 查询问题，例如 "耐心资本的理论机制"
        collection_name: 知识库名称，例如 "nsfc2026"
    """
    if not _initialized:
        return "⏳ 模型正在后台预热（首次启动约需 20-30 秒），请稍后重试。"

    try:
        safe_name  = sanitize_collection_name(collection_name)
        collection = db_client.get_collection(name=safe_name, embedding_function=ef)
    except ValueError as e:
        return f"❌ {e}"
    except Exception:
        return f"❌ 找不到知识库 '{collection_name}'，请确认名称是否正确或先入库。"

    try:
        recall_n = 15 if reranker else 3
        results  = collection.query(query_texts=[query], n_results=recall_n)
        docs     = results["documents"][0]
        metas    = results["metadatas"][0]

        if not docs:
            return "知识库中未找到相关内容。"

        # Rerank 精排（仅当 FlagEmbedding 可用时）
        if reranker and len(docs) > 1:
            scores      = reranker.compute_score([[query, d] for d in docs])
            ranked      = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
            final_docs  = [r[1] for r in ranked[:3]]
            final_metas = [r[2] for r in ranked[:3]]
            header      = f"🎯 Rerank 精排（召回 {len(docs)} 条，返回 Top 3）:\n"
        else:
            final_docs  = docs[:3]
            final_metas = metas[:3]
            header      = f"🔍 向量检索（Top {len(final_docs)}):\n"

        body = "".join(
            f"\n--- 来源: {m.get('source', '未知')} ---\n{d.strip()}\n"
            for d, m in zip(final_docs, final_metas)
        )
        return header + body

    except Exception as e:
        return f"查询出错: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")