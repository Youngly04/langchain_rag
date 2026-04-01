from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from reranker import (
    HuggingFaceReranker,
    RerankCandidate,
    list_supported_rerank_models,
    resolve_rerank_model_name,
)
from utils import load_config


DEFAULT_DATASET_PATH = Path("data/eval/retrieval_eval_dataset.json")
DEFAULT_REPORT_DIR = Path("data/eval/reports")
DEFAULT_CUTOFFS = (1, 3, 5)
DEFAULT_RERANK_MODEL = "bge-base"
DEFAULT_RERANK_CANDIDATES = 10


def load_eval_dataset(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测数据集不存在: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("评测数据集必须是非空列表")

    required_fields = {"id", "question", "expected_sources", "expected_chunk_keywords"}
    for sample in dataset:
        missing = required_fields - set(sample.keys())
        if missing:
            raise ValueError(f"样本缺少字段 {missing}: {sample}")

        if not isinstance(sample["expected_sources"], list) or not sample["expected_sources"]:
            raise ValueError(f"expected_sources 必须是非空列表: {sample}")

        if (
            not isinstance(sample["expected_chunk_keywords"], list)
            or not sample["expected_chunk_keywords"]
        ):
            raise ValueError(f"expected_chunk_keywords 必须是非空列表: {sample}")

    return dataset


def build_vector_store(cfg: dict) -> Chroma:
    embedding = HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model_name"],
        model_kwargs={"device": cfg["embedding"]["device"]},
        encode_kwargs={"normalize_embeddings": cfg["embedding"]["normalize_embeddings"]},
    )

    return Chroma(
        collection_name=cfg["vector_db"]["collection_name"],
        persist_directory=cfg["vector_db"]["persist_directory"],
        embedding_function=embedding,
    )


def build_reranker(model_name: str, device: str) -> HuggingFaceReranker:
    resolved_model_name = resolve_rerank_model_name(model_name)
    return HuggingFaceReranker(model_name=resolved_model_name, device=device)


def get_match_rank(retrieved_sources: list[str], expected_sources: set[str]) -> int | None:
    for idx, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return idx
    return None


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def get_chunk_match_rank(
    retrieved: list[dict],
    expected_sources: set[str],
    expected_chunk_keywords: list[str],
) -> int | None:
    for item in retrieved:
        if item["source"] not in expected_sources:
            continue
        if contains_all_keywords(item["page_content"], expected_chunk_keywords):
            return item["rank"]
    return None


def compute_metrics(details: list[dict], cutoffs: tuple[int, ...], rank_key: str) -> dict:
    total = len(details)
    metrics = {}

    for cutoff in cutoffs:
        hit_count = sum(
            1 for item in details if item[rank_key] is not None and item[rank_key] <= cutoff
        )
        metrics[f"hit@{cutoff}"] = round(hit_count / total, 4)

    mrr = sum(1 / item[rank_key] for item in details if item[rank_key] is not None) / total
    metrics["mrr"] = round(mrr, 4)
    metrics["total_samples"] = total
    metrics["matched_samples"] = sum(1 for item in details if item[rank_key] is not None)
    return metrics


def evaluate_retrieval(
    vector_store: Chroma,
    dataset: list[dict],
    top_k: int,
    reranker: HuggingFaceReranker | None = None,
    rerank_candidates: int | None = None,
) -> list[dict]:
    details = []

    for sample in dataset:
        search_k = rerank_candidates if reranker and rerank_candidates else top_k
        results = vector_store.similarity_search_with_score(sample["question"], k=search_k)

        initial_retrieved = []
        for rank, (doc, score) in enumerate(results, start=1):
            initial_retrieved.append(
                RerankCandidate(
                    source=doc.metadata.get("source"),
                    chunk_id=doc.metadata.get("chunk_id"),
                    page_content=doc.page_content,
                    original_score=round(float(score), 6),
                    original_rank=rank,
                    preview=doc.page_content[:120].replace("\n", " "),
                )
            )

        if reranker is not None:
            reranked_results = reranker.rerank(
                query=sample["question"],
                candidates=initial_retrieved,
                top_n=top_k,
            )
            retrieved = []
            for item in reranked_results:
                retrieved.append(
                    {
                        "rank": item["rank"],
                        "score": round(item["rerank_score"], 6),
                        "source": item["source"],
                        "chunk_id": item["chunk_id"],
                        "preview": item["preview"],
                        "page_content": item["page_content"],
                        "original_rank": item["original_rank"],
                        "original_score": item["original_score"],
                    }
                )
        else:
            retrieved = []
            for candidate in initial_retrieved[:top_k]:
                retrieved.append(
                    {
                        "rank": candidate.original_rank,
                        "score": candidate.original_score,
                        "source": candidate.source,
                        "chunk_id": candidate.chunk_id,
                        "preview": candidate.preview,
                        "page_content": candidate.page_content,
                    }
                )

        retrieved_sources = [item["source"] for item in retrieved if item["source"]]
        expected_sources = set(sample["expected_sources"])
        source_match_rank = get_match_rank(retrieved_sources, expected_sources)
        chunk_match_rank = get_chunk_match_rank(
            retrieved=retrieved,
            expected_sources=expected_sources,
            expected_chunk_keywords=sample["expected_chunk_keywords"],
        )

        report_retrieved = []
        for item in retrieved:
            report_item = {
                "rank": item["rank"],
                "score": item["score"],
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "preview": item["preview"],
            }
            if "original_rank" in item:
                report_item["original_rank"] = item["original_rank"]
                report_item["original_score"] = item["original_score"]
            report_retrieved.append(report_item)

        details.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "expected_sources": sample["expected_sources"],
                "expected_chunk_keywords": sample["expected_chunk_keywords"],
                "source_match_rank": source_match_rank,
                "source_matched": source_match_rank is not None,
                "chunk_match_rank": chunk_match_rank,
                "chunk_matched": chunk_match_rank is not None,
                "notes": sample.get("notes", ""),
                "retrieved": report_retrieved,
            }
        )

    return details


def print_metric_block(title: str, metrics: dict) -> None:
    print(f"\n===== {title} =====")
    print(f"样本数: {metrics['total_samples']}")
    print(f"命中样本数: {metrics['matched_samples']}")
    print(f"Hit@1: {metrics['hit@1']:.4f}")
    print(f"Hit@3: {metrics['hit@3']:.4f}")
    print(f"Hit@5: {metrics['hit@5']:.4f}")
    print(f"MRR:   {metrics['mrr']:.4f}")


def print_summary(source_metrics: dict, chunk_metrics: dict, details: list[dict]) -> None:
    print_metric_block("Source 级命中", source_metrics)
    print_metric_block("Chunk 级命中", chunk_metrics)

    print("\n===== Chunk 未命中样本 =====")
    missed = [item for item in details if not item["chunk_matched"]]
    if not missed:
        print("全部样本都在 top-k 内命中了预期 chunk。")
        return

    for item in missed:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"expected_sources: {item['expected_sources']}")
        print(f"expected_chunk_keywords: {item['expected_chunk_keywords']}")
        print(f"source_match_rank: {item['source_match_rank']}")
        print(f"chunk_match_rank: {item['chunk_match_rank']}")
        for retrieved in item["retrieved"][:3]:
            print(
                f"Top{retrieved['rank']} | source={retrieved['source']} | "
                f"chunk_id={retrieved['chunk_id']} | score={retrieved['score']:.6f}"
            )


def save_report(
    report_dir: Path,
    dataset_path: Path,
    top_k: int,
    cfg: dict,
    rerank_enabled: bool,
    rerank_model: str | None,
    rerank_candidates: int | None,
    source_metrics: dict,
    chunk_metrics: dict,
    details: list[dict],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"retrieval_eval_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "embedding_model": cfg["embedding"]["model_name"],
        "embedding_device": cfg["embedding"]["device"],
        "vector_collection": cfg["vector_db"]["collection_name"],
        "persist_directory": cfg["vector_db"]["persist_directory"],
        "rerank_enabled": rerank_enabled,
        "rerank_model": rerank_model,
        "rerank_model_resolved": resolve_rerank_model_name(rerank_model) if rerank_model else None,
        "rerank_candidates": rerank_candidates,
        "source_metrics": source_metrics,
        "chunk_metrics": chunk_metrics,
        "details": details,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def save_comparison_report(
    report_dir: Path,
    dataset_path: Path,
    top_k: int,
    cfg: dict,
    rerank_candidates: int,
    comparison_results: list[dict],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"retrieval_eval_compare_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "embedding_model": cfg["embedding"]["model_name"],
        "embedding_device": cfg["embedding"]["device"],
        "vector_collection": cfg["vector_db"]["collection_name"],
        "persist_directory": cfg["vector_db"]["persist_directory"],
        "rerank_candidates": rerank_candidates,
        "comparison_results": comparison_results,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def print_supported_rerank_models() -> None:
    print("支持的 rerank 模型别名:")
    for alias, info in list_supported_rerank_models().items():
        print(f"- {alias}: {info['model_name']} | {info['description']}")


def parse_compare_models(compare_value: str) -> list[str]:
    normalized = compare_value.strip()
    if not normalized:
        raise ValueError("compare-rerank-models 不能为空")

    if normalized.lower() == "all":
        return list(list_supported_rerank_models().keys())

    models = [item.strip() for item in normalized.split(",") if item.strip()]
    if not models:
        raise ValueError("compare-rerank-models 至少要提供一个模型别名")

    return models


def evaluate_with_model(
    vector_store: Chroma,
    dataset: list[dict],
    top_k: int,
    rerank_candidates: int,
    cfg: dict,
    model_alias: str,
) -> dict:
    resolved_name = resolve_rerank_model_name(model_alias)
    reranker = build_reranker(model_alias, cfg["embedding"]["device"])
    details = evaluate_retrieval(
        vector_store=vector_store,
        dataset=dataset,
        top_k=top_k,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )
    source_metrics = compute_metrics(details, DEFAULT_CUTOFFS, rank_key="source_match_rank")
    chunk_metrics = compute_metrics(details, DEFAULT_CUTOFFS, rank_key="chunk_match_rank")

    return {
        "label": model_alias,
        "rerank_enabled": True,
        "rerank_model": model_alias,
        "rerank_model_resolved": resolved_name,
        "source_metrics": source_metrics,
        "chunk_metrics": chunk_metrics,
    }


def print_comparison_summary(comparison_results: list[dict]) -> None:
    print("\n===== Rerank 模型对比 =====")
    for item in comparison_results:
        label = item["label"]
        source_metrics = item["source_metrics"]
        chunk_metrics = item["chunk_metrics"]
        print(f"\n[{label}]")
        print(
            "Source | "
            f"Hit@1={source_metrics['hit@1']:.4f} "
            f"Hit@3={source_metrics['hit@3']:.4f} "
            f"Hit@5={source_metrics['hit@5']:.4f} "
            f"MRR={source_metrics['mrr']:.4f}"
        )
        print(
            "Chunk  | "
            f"Hit@1={chunk_metrics['hit@1']:.4f} "
            f"Hit@3={chunk_metrics['hit@3']:.4f} "
            f"Hit@5={chunk_metrics['hit@5']:.4f} "
            f"MRR={chunk_metrics['mrr']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估当前向量检索效果，并支持多 rerank 模型对比")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="评测数据集路径，默认 data/eval/retrieval_eval_dataset.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(DEFAULT_CUTOFFS),
        help="最终保留的检索条数，建议不小于 5",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="评测报告输出目录，默认 data/eval/reports",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="是否启用单模型 rerank",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="rerank 模型别名或 HuggingFace 模型名",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
        help="向量召回后进入 rerank 的候选条数，建议大于等于 top-k",
    )
    parser.add_argument(
        "--list-rerank-models",
        action="store_true",
        help="列出当前内置的 rerank 模型别名后退出",
    )
    parser.add_argument(
        "--compare-rerank-models",
        default=None,
        help="批量对比多个 rerank 模型，传 all 或逗号分隔的模型别名",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_rerank_models:
        print_supported_rerank_models()
        return

    cfg = load_config()
    dataset_path = Path(args.dataset)

    if args.top_k < max(DEFAULT_CUTOFFS):
        raise ValueError("top-k 不能小于 5，否则无法计算 Hit@5")

    if (args.use_rerank or args.compare_rerank_models) and args.rerank_candidates < args.top_k:
        raise ValueError("启用 rerank 时，rerank-candidates 不能小于 top-k")

    persist_directory = Path(cfg["vector_db"]["persist_directory"])
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {persist_directory}。请先运行 build.py 构建向量库。"
        )

    dataset = load_eval_dataset(dataset_path)
    vector_store = build_vector_store(cfg)

    if args.compare_rerank_models:
        model_aliases = parse_compare_models(args.compare_rerank_models)
        comparison_results = []

        baseline_details = evaluate_retrieval(
            vector_store=vector_store,
            dataset=dataset,
            top_k=args.top_k,
            reranker=None,
            rerank_candidates=None,
        )
        comparison_results.append(
            {
                "label": "baseline",
                "rerank_enabled": False,
                "rerank_model": None,
                "rerank_model_resolved": None,
                "source_metrics": compute_metrics(
                    baseline_details, DEFAULT_CUTOFFS, rank_key="source_match_rank"
                ),
                "chunk_metrics": compute_metrics(
                    baseline_details, DEFAULT_CUTOFFS, rank_key="chunk_match_rank"
                ),
            }
        )

        for model_alias in model_aliases:
            comparison_results.append(
                evaluate_with_model(
                    vector_store=vector_store,
                    dataset=dataset,
                    top_k=args.top_k,
                    rerank_candidates=args.rerank_candidates,
                    cfg=cfg,
                    model_alias=model_alias,
                )
            )

        report_path = save_comparison_report(
            report_dir=Path(args.report_dir),
            dataset_path=dataset_path,
            top_k=args.top_k,
            cfg=cfg,
            rerank_candidates=args.rerank_candidates,
            comparison_results=comparison_results,
        )
        print_comparison_summary(comparison_results)
        print(f"\n对比报告已保存到: {report_path}")
        return

    reranker = None
    if args.use_rerank:
        reranker = build_reranker(
            model_name=args.rerank_model,
            device=cfg["embedding"]["device"],
        )

    details = evaluate_retrieval(
        vector_store=vector_store,
        dataset=dataset,
        top_k=args.top_k,
        reranker=reranker,
        rerank_candidates=args.rerank_candidates,
    )
    source_metrics = compute_metrics(details, DEFAULT_CUTOFFS, rank_key="source_match_rank")
    chunk_metrics = compute_metrics(details, DEFAULT_CUTOFFS, rank_key="chunk_match_rank")
    report_path = save_report(
        report_dir=Path(args.report_dir),
        dataset_path=dataset_path,
        top_k=args.top_k,
        cfg=cfg,
        rerank_enabled=args.use_rerank,
        rerank_model=args.rerank_model if args.use_rerank else None,
        rerank_candidates=args.rerank_candidates if args.use_rerank else None,
        source_metrics=source_metrics,
        chunk_metrics=chunk_metrics,
        details=details,
    )

    print_summary(source_metrics, chunk_metrics, details)
    print(f"\n评测报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
