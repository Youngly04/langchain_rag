from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from reranker import HuggingFaceReranker, RerankCandidate, resolve_rerank_model_name
from utils import load_config


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_rerank: bool | None = Field(default=None)
    rerank_candidates: int | None = Field(default=None, ge=1, le=50)


class ReferenceItem(BaseModel):
    rank: int
    score: float
    source: str | None
    chunk_id: int | None


class ChatResponse(BaseModel):
    question: str
    answer: str
    references: list[ReferenceItem]


class RAGService:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.embedding: HuggingFaceEmbeddings | None = None
        self.vector_store: Chroma | None = None
        self.llm: ChatOpenAI | None = None
        self.reranker: HuggingFaceReranker | None = None

        self.default_top_k = self.cfg["retrieval"]["top_k"]
        self.rerank_enabled = True
        self.rerank_model = "bce-base"
        self.rerank_candidates = 10

    def initialize(self) -> None:
        embed_model_name = self.cfg["embedding"]["model_name"]
        embed_device = self.cfg["embedding"]["device"]
        normalize_embeddings = self.cfg["embedding"]["normalize_embeddings"]

        self.embedding = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": embed_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )

        self.vector_store = Chroma(
            collection_name=self.cfg["vector_db"]["collection_name"],
            persist_directory=self.cfg["vector_db"]["persist_directory"],
            embedding_function=self.embedding,
        )

        self.llm = ChatOpenAI(
            model=self.cfg["llm"]["model_name"],
            api_key=self.cfg["llm"]["api_key"],
            base_url=self.cfg["llm"]["base_url"],
            temperature=0,
        )

        if self.rerank_enabled:
            self.reranker = HuggingFaceReranker(
                model_name=resolve_rerank_model_name(self.rerank_model),
                device=embed_device,
            )

    def ensure_ready(self) -> None:
        if self.vector_store is None or self.llm is None:
            raise RuntimeError("RAG service is not initialized")

    @staticmethod
    def format_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
        context_parts = []
        references = []

        for i, item in enumerate(retrieved, start=1):
            source = item.get("source")
            chunk_id = item.get("chunk_id")
            score = float(item.get("rerank_score", item.get("original_score", 0.0)))

            references.append(
                {
                    "rank": i,
                    "score": score,
                    "source": source,
                    "chunk_id": chunk_id,
                }
            )

            context_parts.append(
                f"[参考片段{i} | source={source} | chunk_id={chunk_id}]\n{item['page_content']}"
            )

        return "\n\n--------------------\n\n".join(context_parts), references

    def retrieve(
        self,
        question: str,
        top_k: int,
        use_rerank: bool,
        rerank_candidates: int,
    ) -> list[dict]:
        self.ensure_ready()

        if use_rerank and rerank_candidates < top_k:
            raise ValueError("rerank_candidates cannot be smaller than top_k")

        search_k = rerank_candidates if use_rerank and self.reranker else top_k
        results = self.vector_store.similarity_search_with_score(question, k=search_k)

        initial_retrieved = []
        for rank, (doc, score) in enumerate(results, start=1):
            initial_retrieved.append(
                RerankCandidate(
                    source=doc.metadata.get("source"),
                    chunk_id=doc.metadata.get("chunk_id"),
                    page_content=doc.page_content,
                    original_score=float(score),
                    original_rank=rank,
                    preview=doc.page_content[:120].replace("\n", " "),
                )
            )

        if use_rerank and self.reranker:
            return self.reranker.rerank(
                query=question,
                candidates=initial_retrieved,
                top_n=top_k,
            )

        return [
            {
                "rank": candidate.original_rank,
                "source": candidate.source,
                "chunk_id": candidate.chunk_id,
                "page_content": candidate.page_content,
                "original_score": candidate.original_score,
                "original_rank": candidate.original_rank,
            }
            for candidate in initial_retrieved[:top_k]
        ]

    @staticmethod
    def build_prompt(question: str, context_text: str) -> str:
        return f"""
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
3. 如有必要，可以分点回答，但不要为了分点而分点。
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

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        rerank_candidates: int | None = None,
    ) -> dict:
        self.ensure_ready()

        resolved_top_k = top_k or self.default_top_k
        resolved_use_rerank = self.rerank_enabled if use_rerank is None else use_rerank
        resolved_rerank_candidates = rerank_candidates or self.rerank_candidates

        retrieved = self.retrieve(
            question=question,
            top_k=resolved_top_k,
            use_rerank=resolved_use_rerank,
            rerank_candidates=resolved_rerank_candidates,
        )

        if not retrieved:
            return {
                "question": question,
                "answer": "当前没有检索到相关内容，暂时无法回答这个问题。",
                "references": [],
            }

        context_text, references = self.format_context(retrieved)
        prompt = self.build_prompt(question, context_text)
        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "references": references,
        }


rag_service = RAGService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    rag_service.initialize()
    yield


app = FastAPI(title="LangChain RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_service.answer(
            question=request.question.strip(),
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            rerank_candidates=request.rerank_candidates,
        )
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
