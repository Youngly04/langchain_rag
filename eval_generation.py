from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from generation_evaluator import (
    RuleEvaluationResult,
    build_judge_prompt,
    evaluate_rules,
    judge_average_score,
    parse_judge_response,
)
from utils import load_config


DEFAULT_DATASET_PATH = Path("data/eval/generation_eval_dataset.json")
DEFAULT_REPORT_DIR = Path("data/eval/reports")
DEFAULT_TOP_K = 4


def load_eval_dataset(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"回答评测数据集不存在: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("回答评测数据集必须是非空列表")

    required_fields = {
        "id",
        "question",
        "expected_sources",
        "reference_answer",
        "must_include",
        "should_include",
        "must_not_include",
    }
    for sample in dataset:
        missing = required_fields - set(sample.keys())
        if missing:
            raise ValueError(f"样本缺少字段 {missing}: {sample}")

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


def build_llm(cfg: dict) -> ChatOpenAI:
    llm_api_key = cfg["llm"]["api_key"]
    if not llm_api_key or llm_api_key == "xxx":
        raise ValueError("请先在 config/config.yaml 中填写可用的 LLM API Key")

    return ChatOpenAI(
        model=cfg["llm"]["model_name"],
        api_key=llm_api_key,
        base_url=cfg["llm"]["base_url"],
        temperature=0,
    )


def format_context(results: list[tuple]) -> tuple[str, list[dict]]:
    context_parts = []
    references = []

    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source")
        chunk_id = doc.metadata.get("chunk_id")
        references.append(
            {
                "rank": idx,
                "score": round(float(score), 6),
                "source": source,
                "chunk_id": chunk_id,
                "content": doc.page_content,
            }
        )
        context_parts.append(
            f"[参考片段{idx} | source={source} | chunk_id={chunk_id}]\n{doc.page_content}"
        )

    return "\n\n--------------------\n\n".join(context_parts), references


def build_answer_prompt(question: str, context_text: str) -> str:
    return f"""
你是一名电商售后客服知识库助手，负责根据知识库内容回答用户关于退款、退货、换货、维修、质保、运费等问题。

请严格依据“参考片段”作答，并遵守以下要求：
1. 优先依据参考片段直接回答，不要编造资料中没有的信息。
2. 如果资料不足以确认，请明确说“根据当前知识库，暂时无法确认”。
3. 回答要像真实售后客服，语气自然、礼貌、清晰，尽量先给结论，再补充说明。
4. 如果涉及操作路径、条件限制、赔付规则，请只说参考片段里明确提到的内容。
5. 不要擅自补充到账时间、赔付比例、处理时效、额外渠道等资料未明确给出的信息。
6. 最后一行单独输出参考依据，格式必须为：
参考依据：source=..., chunk_id=...

用户问题：
{question}

参考片段：
{context_text}

请开始回答：
""".strip()


def evaluate_sample(
    sample: dict,
    vector_store: Chroma,
    llm: ChatOpenAI,
    top_k: int,
) -> dict:
    results = vector_store.similarity_search_with_score(sample["question"], k=top_k)
    context_text, references = format_context(results)
    answer_prompt = build_answer_prompt(sample["question"], context_text)
    answer_response = llm.invoke(answer_prompt)
    answer = answer_response.content.strip()

    retrieved_sources = [item["source"] for item in references if item["source"]]
    rule_result: RuleEvaluationResult = evaluate_rules(
        answer=answer,
        expected_sources=sample["expected_sources"],
        must_include=sample["must_include"],
        should_include=sample["should_include"],
        must_not_include=sample["must_not_include"],
        retrieved_sources=retrieved_sources,
    )

    judge_prompt = build_judge_prompt(
        question=sample["question"],
        reference_answer=sample["reference_answer"],
        answer=answer,
        retrieved_context=context_text,
        expected_sources=sample["expected_sources"],
    )
    judge_response = llm.invoke(judge_prompt)
    judge_result = parse_judge_response(judge_response.content.strip())

    return {
        "id": sample["id"],
        "scenario": sample.get("scenario", ""),
        "question": sample["question"],
        "expected_sources": sample["expected_sources"],
        "reference_answer": sample["reference_answer"],
        "answer": answer,
        "rules": {
            "score": rule_result.score,
            "passed": rule_result.passed,
            "source_expected_hit": rule_result.source_expected_hit,
            "source_cited_hit": rule_result.source_cited_hit,
            "reference_format_ok": rule_result.reference_format_ok,
            "must_include_recall": rule_result.must_include_recall,
            "should_include_recall": rule_result.should_include_recall,
            "forbidden_violations": rule_result.forbidden_violations,
            "missing_must_include": rule_result.missing_must_include,
            "missing_should_include": rule_result.missing_should_include,
            "violated_forbidden": rule_result.violated_forbidden,
            "cited_sources": rule_result.cited_sources,
            "cited_chunk_ids": rule_result.cited_chunk_ids,
        },
        "judge": {
            "correctness": judge_result.correctness,
            "groundedness": judge_result.groundedness,
            "completeness": judge_result.completeness,
            "customer_service_tone": judge_result.customer_service_tone,
            "no_hallucination": judge_result.no_hallucination,
            "overall": judge_result.overall,
            "average_score": judge_average_score(judge_result),
            "passed": judge_result.passed,
            "summary": judge_result.summary,
            "issues": judge_result.issues,
            "raw_text": judge_result.raw_text,
        },
        "retrieved": [
            {
                "rank": item["rank"],
                "score": item["score"],
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "preview": item["content"][:120].replace("\n", " "),
            }
            for item in references
        ],
        "notes": sample.get("notes", ""),
    }


def aggregate_results(details: list[dict]) -> dict:
    total = len(details)
    rule_passed = sum(1 for item in details if item["rules"]["passed"])
    judge_passed = sum(1 for item in details if item["judge"]["passed"])

    avg_rule_score = round(sum(item["rules"]["score"] for item in details) / total, 4)
    avg_judge_score = round(sum(item["judge"]["average_score"] for item in details) / total, 4)
    avg_correctness = round(sum(item["judge"]["correctness"] for item in details) / total, 4)
    avg_groundedness = round(sum(item["judge"]["groundedness"] for item in details) / total, 4)
    avg_completeness = round(sum(item["judge"]["completeness"] for item in details) / total, 4)
    avg_tone = round(sum(item["judge"]["customer_service_tone"] for item in details) / total, 4)
    avg_no_hallucination = round(
        sum(item["judge"]["no_hallucination"] for item in details) / total, 4
    )

    return {
        "total_samples": total,
        "rule_pass_rate": round(rule_passed / total, 4),
        "judge_pass_rate": round(judge_passed / total, 4),
        "avg_rule_score": avg_rule_score,
        "avg_judge_score": avg_judge_score,
        "avg_correctness": avg_correctness,
        "avg_groundedness": avg_groundedness,
        "avg_completeness": avg_completeness,
        "avg_customer_service_tone": avg_tone,
        "avg_no_hallucination": avg_no_hallucination,
    }


def print_summary(summary: dict, details: list[dict]) -> None:
    print("\n===== 回答评测汇总 =====")
    print(f"样本数: {summary['total_samples']}")
    print(f"规则通过率: {summary['rule_pass_rate']:.4f}")
    print(f"Judge 通过率: {summary['judge_pass_rate']:.4f}")
    print(f"规则平均分: {summary['avg_rule_score']:.4f}")
    print(f"Judge 平均分: {summary['avg_judge_score']:.4f}")
    print(f"准确性均分: {summary['avg_correctness']:.4f}")
    print(f"基于资料作答均分: {summary['avg_groundedness']:.4f}")
    print(f"完整性均分: {summary['avg_completeness']:.4f}")
    print(f"客服表达均分: {summary['avg_customer_service_tone']:.4f}")
    print(f"无幻觉均分: {summary['avg_no_hallucination']:.4f}")

    weak_cases = [
        item
        for item in details
        if (not item["rules"]["passed"]) or item["judge"]["average_score"] < 4.5
    ]
    print("\n===== 需要重点关注的样本 =====")
    if not weak_cases:
        print("当前没有低分样本。")
        return

    for item in weak_cases[:8]:
        print(f"\n[{item['id']}] {item['question']}")
        print(
            f"rule_score={item['rules']['score']:.4f} | "
            f"judge_avg={item['judge']['average_score']:.4f}"
        )
        print(f"judge_summary={item['judge']['summary']}")
        if item["judge"]["issues"]:
            print(f"judge_issues={item['judge']['issues']}")


def save_report(
    report_dir: Path,
    dataset_path: Path,
    cfg: dict,
    top_k: int,
    details: list[dict],
    summary: dict,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"generation_eval_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "embedding_model": cfg["embedding"]["model_name"],
        "embedding_device": cfg["embedding"]["device"],
        "llm_model": cfg["llm"]["model_name"],
        "vector_collection": cfg["vector_db"]["collection_name"],
        "persist_directory": cfg["vector_db"]["persist_directory"],
        "summary": summary,
        "details": details,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 RAG 客服回答质量")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="回答评测数据集路径，默认 data/eval/generation_eval_dataset.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="生成答案前使用的检索条数，默认 4",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="报告输出目录，默认 data/eval/reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    dataset_path = Path(args.dataset)
    persist_directory = Path(cfg["vector_db"]["persist_directory"])

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {persist_directory}。请先运行 build.py 构建向量库。"
        )

    dataset = load_eval_dataset(dataset_path)
    vector_store = build_vector_store(cfg)
    llm = build_llm(cfg)

    details = []
    for sample in dataset:
        details.append(
            evaluate_sample(
                sample=sample,
                vector_store=vector_store,
                llm=llm,
                top_k=args.top_k,
            )
        )

    summary = aggregate_results(details)
    report_path = save_report(
        report_dir=Path(args.report_dir),
        dataset_path=dataset_path,
        cfg=cfg,
        top_k=args.top_k,
        details=details,
        summary=summary,
    )

    print_summary(summary, details)
    print(f"\n回答评测报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
