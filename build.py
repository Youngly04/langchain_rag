from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from utils import load_config


def load_markdown_docs(raw_dir: Path):
    docs = []
    for path in raw_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name}
            )
        )
    return docs


def main():
    cfg = load_config()

    raw_dir = Path(cfg["crawler"]["save_dir"])
    persist_directory = cfg["vector_db"]["persist_directory"]
    collection_name = cfg["vector_db"]["collection_name"]

    embed_model_name = cfg["embedding"]["model_name"]
    embed_device = cfg["embedding"]["device"]
    normalize_embeddings = cfg["embedding"]["normalize_embeddings"]

    chunk_size = cfg["splitter"]["chunk_size"]
    chunk_overlap = cfg["splitter"]["chunk_overlap"]
    separators = cfg["splitter"]["separators"]

    docs = load_markdown_docs(raw_dir)
    if not docs:
        raise ValueError("data/raw 目录下没有可用文档，请先运行 crawler.py。")

    print(f"原始文档数: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"切块后数量: {len(chunks)}")

    db_path = Path(persist_directory)
    if db_path.exists():
        shutil.rmtree(db_path)
        print(f"已删除旧向量库: {db_path}")

    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"向量库已构建完成: {persist_directory}")
    print(f"collection_name: {collection_name}")

    print("\n===== 示例切块 =====")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n[Chunk {i}]")
        print(f"source={chunk.metadata.get('source')}, chunk_id={chunk.metadata.get('chunk_id')}")
        print(chunk.page_content[:300])


if __name__ == "__main__":
    main()