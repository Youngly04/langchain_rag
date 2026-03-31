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
        你是一名电商售后客服知识库助手，负责根据知识库内容回答用户关于退货、换货、退款、维修、质保、运费等问题。

        请严格依据“参考片段”回答，要求如下：

        【回答原则】
        1. 只能优先依据参考片段作答，不要编造参考片段中没有的信息。
        2. 如果参考片段已经可以回答问题，就直接给出明确结论，不要模糊表达。
        3. 如果参考片段信息不足，请明确说：“根据当前知识库，暂时无法确认。”
        4. 如果多个参考片段内容可以互相补充，可以整合后回答；如果存在冲突，优先采用更直接、更明确、与用户问题更相关的内容。
        5. 不要输出“根据参考片段”“检索结果显示”“Top1/Top2”之类的话。

        【表达要求】
        1. 语气自然、礼貌、简洁，像真实售后客服，不要写成论文总结或机械罗列。
        2. 优先采用“先结论，后说明”的方式回答。
        3. 如果合适，可以分点回答，但不要为了分点而分点。
        4. 能直接告诉用户怎么做，就直接说清楚操作方式或处理路径。
        5. 不要输出知识库中没有明确提到的承诺性内容。

        【输出格式】
        请尽量按下面结构回答：
        1. 先用1到2句话直接回答用户问题；
        2. 如有必要，再补充操作步骤、适用条件或注意事项；
        3. 最后一行单独输出参考依据，格式必须为：
        参考依据：source=..., chunk_id=...

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