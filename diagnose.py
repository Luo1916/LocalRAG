"""
baai_rag 诊断脚本 - 逐步测试每个加载步骤，精确定位卡死位置
运行方法: python diagnose.py
"""
import os
import sys
import time

BASE_DIR = r"D:\baai_rag"
DB_PATH  = os.path.join(BASE_DIR, "chroma_db_data")
os.environ["HF_HOME"]                 = os.path.join(BASE_DIR, "hf_models")
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def step(msg):
    print(f"\n{'='*60}")
    print(f"  步骤: {msg}")
    print(f"{'='*60}")
    return time.time()

def done(t0, ok=True):
    elapsed = time.time() - t0
    status = "✅ 成功" if ok else "❌ 失败"
    print(f"  {status}  耗时: {elapsed:.1f}s")

# ── 步骤 1: 检查 torch ──────────────────────────────────────────
t = step("导入 torch 并检测 CUDA")
try:
    import torch
    print(f"  torch 版本: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  当前已占用显存: {mem_alloc:.2f} GB")
    else:
        print("  未检测到 CUDA，将使用 CPU")
    done(t)
except Exception as e:
    done(t, ok=False)
    print(f"  错误信息: {e}")
    sys.exit(1)

# ── 步骤 2: 检查 chromadb ──────────────────────────────────────
t = step("导入 chromadb 并连接数据库")
try:
    import chromadb
    from chromadb.utils import embedding_functions
    print(f"  chromadb 版本: {chromadb.__version__}")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    cols = db_client.list_collections()
    print(f"  DB路径: {DB_PATH}")
    print(f"  现有知识库: {[c.name for c in cols] if cols else '（空）'}")
    done(t)
except Exception as e:
    done(t, ok=False)
    print(f"  错误信息: {e}")
    sys.exit(1)

# ── 步骤 3: 加载 bge-m3 向量模型 ──────────────────────────────
t = step(f"加载 BAAI/bge-m3（device={device}）")
try:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3",
        device=device
    )
    done(t)
except Exception as e:
    done(t, ok=False)
    print(f"  错误信息: {e}")
    ef = None

# ── 步骤 4: Embedding Warmup ─────────────────────────────────
if ef:
    t = step("Embedding Warmup（首次推理）")
    try:
        result = ef(["warmup test"])
        print(f"  向量维度: {len(result[0])}")
        done(t)
    except Exception as e:
        done(t, ok=False)
        print(f"  错误信息: {e}")

# ── 步骤 5: 加载 FlagReranker ─────────────────────────────────
t = step("加载 BAAI/bge-reranker-m3（FlagEmbedding）")
try:
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=(device == "cuda"))
    done(t)
except ImportError:
    done(t, ok=False)
    print("  ⚠️ FlagEmbedding 未安装，Reranker 跳过（不影响主功能）")
    reranker = None
except Exception as e:
    done(t, ok=False)
    print(f"  错误信息: {e}")
    reranker = None

# ── 步骤 6: Reranker Warmup ──────────────────────────────────
if reranker:
    t = step("Reranker Warmup（首次评分）")
    try:
        scores = reranker.compute_score([["测试问题", "测试答案"]])
        print(f"  得分: {scores}")
        done(t)
    except Exception as e:
        done(t, ok=False)
        print(f"  错误信息: {e}")

# ── 步骤 7: 检查显存使用 ─────────────────────────────────────
if cuda_ok:
    t = step("最终显存状态")
    try:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  已分配: {allocated:.2f} GB / 总计: {total:.1f} GB")
        print(f"  已预留: {reserved:.2f} GB")
        print(f"  剩余可用: {total - reserved:.2f} GB")
        done(t)
    except Exception as e:
        done(t, ok=False)

print("\n" + "="*60)
print("  🎉 诊断完成！所有步骤均已完成，请查看上方各步骤耗时。")
print("="*60)
