from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass


REFERENCE_PATTERN = re.compile(r"参考依据：(.+)$", re.MULTILINE)
SOURCE_PATTERN = re.compile(r"source=([^,\n]+)")
CHUNK_ID_PATTERN = re.compile(r"chunk_id=([0-9]+)")


@dataclass
class RuleEvaluationResult:
    score: float
    source_expected_hit: bool
    source_cited_hit: bool
    reference_format_ok: bool
    must_include_recall: float
    should_include_recall: float
    forbidden_violations: int
    passed: bool
    missing_must_include: list[str]
    missing_should_include: list[str]
    violated_forbidden: list[str]
    cited_sources: list[str]
    cited_chunk_ids: list[int]


@dataclass
class JudgeEvaluationResult:
    correctness: int
    groundedness: int
    completeness: int
    customer_service_tone: int
    no_hallucination: int
    overall: int
    passed: bool
    summary: str
    issues: list[str]
    raw_text: str


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\ufeff", "")
    return re.sub(r"\s+", "", normalized)


def normalize_source_name(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("\\", "/")
    normalized = normalized.rstrip(".。；;，,")
    return normalized


def source_matches(candidate: str, expected_sources: list[str]) -> bool:
    normalized_candidate = normalize_source_name(candidate)
    normalized_expected = {normalize_source_name(item) for item in expected_sources}
    if normalized_candidate in normalized_expected:
        return True

    candidate_core = normalized_candidate.removesuffix(".md")
    expected_cores = {item.removesuffix(".md") for item in normalized_expected}
    if candidate_core in expected_cores:
        return True

    return any(
        candidate_core in expected_core or expected_core in candidate_core
        for expected_core in expected_cores
        if candidate_core and expected_core
    )


def compute_keyword_recall(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 1.0, []

    normalized_answer = normalize_text(answer)
    missing = [keyword for keyword in keywords if normalize_text(keyword) not in normalized_answer]
    matched = len(keywords) - len(missing)
    return round(matched / len(keywords), 4), missing


def find_forbidden_keywords(answer: str, keywords: list[str]) -> list[str]:
    normalized_answer = normalize_text(answer)
    return [keyword for keyword in keywords if normalize_text(keyword) in normalized_answer]


def extract_reference_line(answer: str) -> str | None:
    match = REFERENCE_PATTERN.search(answer or "")
    if not match:
        return None
    return match.group(1).strip()


def extract_citations(answer: str) -> tuple[list[str], list[int], bool]:
    reference_line = extract_reference_line(answer)
    if not reference_line:
        return [], [], False

    sources = [item.strip() for item in SOURCE_PATTERN.findall(reference_line)]
    chunk_ids = [int(item) for item in CHUNK_ID_PATTERN.findall(reference_line)]
    return sources, chunk_ids, True


def evaluate_rules(
    answer: str,
    expected_sources: list[str],
    must_include: list[str],
    should_include: list[str],
    must_not_include: list[str],
    retrieved_sources: list[str],
) -> RuleEvaluationResult:
    cited_sources, cited_chunk_ids, reference_format_ok = extract_citations(answer)

    must_include_recall, missing_must_include = compute_keyword_recall(answer, must_include)
    should_include_recall, missing_should_include = compute_keyword_recall(answer, should_include)
    violated_forbidden = find_forbidden_keywords(answer, must_not_include)

    source_expected_hit = any(source_matches(source, expected_sources) for source in retrieved_sources)
    source_cited_hit = any(source_matches(source, expected_sources) for source in cited_sources)

    score = 0.0
    score += 0.20 if source_expected_hit else 0.0
    score += 0.15 if source_cited_hit else 0.0
    score += 0.10 if reference_format_ok else 0.0
    score += 0.35 * must_include_recall
    score += 0.10 * should_include_recall
    if violated_forbidden:
        score -= min(0.30, 0.10 * len(violated_forbidden))

    score = max(0.0, min(1.0, round(score, 4)))
    passed = (
        must_include_recall >= 0.8
        and not violated_forbidden
        and source_expected_hit
        and source_cited_hit
        and reference_format_ok
    )

    return RuleEvaluationResult(
        score=score,
        source_expected_hit=source_expected_hit,
        source_cited_hit=source_cited_hit,
        reference_format_ok=reference_format_ok,
        must_include_recall=must_include_recall,
        should_include_recall=should_include_recall,
        forbidden_violations=len(violated_forbidden),
        passed=passed,
        missing_must_include=missing_must_include,
        missing_should_include=missing_should_include,
        violated_forbidden=violated_forbidden,
        cited_sources=cited_sources,
        cited_chunk_ids=cited_chunk_ids,
    )


def build_judge_prompt(
    question: str,
    reference_answer: str,
    answer: str,
    retrieved_context: str,
    expected_sources: list[str],
) -> str:
    return f"""
你是一名严格、挑错优先的 RAG 客服回答评审员。

请根据“用户问题”“参考答案”“检索上下文”“模型回答”进行评分。
你的职责不是帮回答找优点，而是优先找错误、遗漏、越界承诺、幻觉和客服表达不当之处。
如果回答存在任何资料未明确支持的时间承诺、赔付标准、处理路径、绝对化保证，必须扣分。
如果回答虽然大方向正确，但遗漏关键条件、限制、例外情况，也必须扣分。
除非回答非常准确、完整、严格基于资料、且表达符合客服场景，否则不要给 5 分。

请按以下维度评分：
1. correctness：是否与参考答案和上下文一致。
2. groundedness：是否严格基于检索上下文，没有超出资料乱说。
3. completeness：关键条件、限制、处理方式是否说全。
4. customer_service_tone：是否像真实客服，清晰、稳妥、礼貌，不过度承诺。
5. no_hallucination：是否避免编造资料未明确给出的事实。
6. overall：综合评分，不能高于最明显短板太多。

评分标准：
- 5 分：几乎无可挑剔，准确完整，严格基于资料，没有明显遗漏和越界。
- 4 分：总体较好，但有轻微遗漏或表达可以更稳妥。
- 3 分：基本可用，但有明显遗漏、表达不严谨或轻微越界。
- 2 分：有明显错误、关键遗漏或较明显的资料外推断。
- 1 分：严重错误、明显幻觉、与资料冲突，或极不符合客服场景。

输出必须是 JSON，对象字段固定如下：
{{
  "correctness": 1-5,
  "groundedness": 1-5,
  "completeness": 1-5,
  "customer_service_tone": 1-5,
  "no_hallucination": 1-5,
  "overall": 1-5,
  "passed": true/false,
  "summary": "一句中文总结",
  "issues": ["问题1", "问题2"]
}}

补充要求：
- 如果给出 5 分，issues 必须为空，且 summary 中应明确说明为什么接近满分。
- 如果存在资料外承诺或关键遗漏，passed 必须是 false。
- issues 至少列出 0-3 条最关键问题；没有问题才返回空列表。

用户问题：
{question}

期望来源：
{", ".join(expected_sources)}

参考答案：
{reference_answer}

检索上下文：
{retrieved_context}

模型回答：
{answer}
""".strip()


def extract_json_object(text: str) -> dict:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Judge 输出为空")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(stripped[idx:])
            return data
        except json.JSONDecodeError:
            continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge 输出中未找到有效 JSON 对象")

    return json.loads(stripped[start : end + 1])


def parse_judge_response(raw_text: str) -> JudgeEvaluationResult:
    data = extract_json_object(raw_text)

    required_fields = {
        "correctness",
        "groundedness",
        "completeness",
        "customer_service_tone",
        "no_hallucination",
        "overall",
        "passed",
        "summary",
        "issues",
    }
    missing = required_fields - set(data.keys())
    if missing:
        raise ValueError(f"Judge 输出缺少字段: {missing}")

    result = JudgeEvaluationResult(
        correctness=int(data["correctness"]),
        groundedness=int(data["groundedness"]),
        completeness=int(data["completeness"]),
        customer_service_tone=int(data["customer_service_tone"]),
        no_hallucination=int(data["no_hallucination"]),
        overall=int(data["overall"]),
        passed=bool(data["passed"]),
        summary=str(data["summary"]),
        issues=[str(item) for item in data["issues"]],
        raw_text=raw_text,
    )

    hard_fail = min(
        result.correctness,
        result.groundedness,
        result.no_hallucination,
    ) <= 3
    if hard_fail or result.overall <= 3:
        result.passed = False

    if result.overall == 5 and result.issues:
        result.passed = False

    return result


def judge_average_score(result: JudgeEvaluationResult) -> float:
    values = [
        result.correctness,
        result.groundedness,
        result.completeness,
        result.customer_service_tone,
        result.no_hallucination,
        result.overall,
    ]
    return round(sum(values) / len(values), 4)
