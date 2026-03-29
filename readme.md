# LangChain RAG

## 环境安装

```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
```

## 配置说明

运行前需要先修改 `config/config.yaml`，主要填写两部分内容：

- `llm.api_key`：
- `base_url`：

## 运行步骤

### 1. 爬取网页内容

```bash
python crawler.py
```
运行后会在 `data/raw/` 目录下生成原始文本文件。

### 2. 构建向量库

```bash
python build.py
```
运行后会完成以下流程：

- 读取原始文本
- 文本分块
- embedding 向量化
- 构建 Chroma 向量库

向量库会保存在 `data/chroma_db/` 目录下。

### 3. 启动 RAG 问答

```bash
python rag.py
```

启动后可以直接在终端输入问题，例如：

```bash
退款会退到哪里
如何申请售后
公司转账订单退款怎么退
京东E卡退款可以提现吗
```

输入 `q` 可以退出程序。
