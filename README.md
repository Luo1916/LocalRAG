# LocalRAG

基于 BAAI bge-m3 向量模型和 MCP 协议的本地知识库检索系统，专为 AI 助手（如 Claude Desktop）设计。

## 📋 系统要求
- Python 3.10+
- 支持 CUDA 的 GPU (可选，推荐 NVIDIA RTX 系列)

## 🚀 快速开始

### 1. 安装基础依赖
```bash
pip install -r requirements.txt
```

### 2. GPU 硬件加速 (强烈推荐)
如果你拥有 NVIDIA 独立显卡，请额外安装支持 CUDA 的 PyTorch (以 CUDA 12.1 为例) 以启用推理加速：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 运行服务器
```bash
python baai_rag_server.py
```
*首次启动时，程序会自动从 Hugging Face 下载模型文件（约 2GB），请保持网络畅通。*

### 4. Claude Desktop 配置
在 Claude Desktop 的 `claude_desktop_config.json` 中配置：
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
重启 Claude Desktop 即可。