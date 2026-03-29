from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from utils import load_config


def format_context(results):
    context_parts = []
    references = []

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source")
        chunk_id = doc.metadata.get("chunk_id")
        references.append(
            {
                "rank": i,
                "score": score,
                "source": source,
                "chunk_id": chunk_id,
            }
        )

        context_parts.append(
            f"[参考片段 {i} | source={source} | chunk_id={chunk_id}]\n{doc.page_content}"
        )

    return "\n\n--------------------\n\n".join(context_parts), references


def main():
    cfg = load_config()

    persist_directory = cfg["vector_db"]["persist_directory"]
    collection_name = cfg["vector_db"]["collection_name"]

    embed_model_name = cfg["embedding"]["model_name"]
    embed_device = cfg["embedding"]["device"]
    normalize_embeddings = cfg["embedding"]["normalize_embeddings"]

    top_k = cfg["retrieval"]["top_k"]

    llm_model_name = cfg["llm"]["model_name"]
    llm_api_key = cfg["llm"]["api_key"]
    llm_base_url = cfg["llm"]["base_url"]

    if not llm_api_key or llm_api_key == "你的智谱API Key":
        raise ValueError("请先在 config/config.yaml 中填写正确的智谱 API Key。")

    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding,
    )

    llm = ChatOpenAI(
        model=llm_model_name,
        api_key=llm_api_key,
        base_url=llm_base_url,
        temperature=0,
    )

    print("RAG 问答已启动，输入 q 退出。")

    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() == "q":
            print("已退出。")
            break

        if not question:
            print("问题不能为空，请重新输入。")
            continue

        results = vector_store.similarity_search_with_score(question, k=top_k)

        if not results:
            print("没有检索到结果。")
            continue

        context_text, references = format_context(results)

        prompt = f"""
你是一个电商售后客服知识库助手。
请严格根据“参考片段”回答用户问题，要求如下：

1. 优先依据参考片段作答，不要脱离参考片段自由发挥。
2. 回答要简洁、清楚、像客服说话。
3. 若参考片段信息不足，请明确说“根据当前知识库，暂时无法确认”。
4. 尽量用分点形式回答。
5. 回答结束后，补一句“参考依据：source=..., chunk_id=...”。

用户问题：
{question}

参考片段：
{context_text}

请开始回答：
""".strip()

        response = llm.invoke(prompt)

        print("\n===== 最终回答 =====")
        print(response.content)

        print("\n===== 检索参考 =====")
        for ref in references:
            print(
                f"Top{ref['rank']} | score={ref['score']:.6f} | "
                f"source={ref['source']} | chunk_id={ref['chunk_id']}"
            )


if __name__ == "__main__":
    main()