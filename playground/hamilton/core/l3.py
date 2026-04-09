"""Hamilton L3 memory store.

The L3 layer is a persistent, cross-task experience store. It only keeps
distilled records that can transfer across tasks instead of raw trajectories.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    CRITIC_CHALLENGE_TYPES,
    CRITIC_CONTEXT_FILE,
    DEBATE_STATE_FILE,
    HCC_LEDGER_FILE,
    L3_CONTEXT_FILE,
    L3_HITS_FILE,
    L3_INDEX_FILE,
    L3_TASKS_DIR,
    TASK_SIGNATURE_FILE,
)

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
CSV_RE = re.compile(r"[\w./-]+\.csv", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "任务",
    "数据",
    "发现",
    "结果",
    "当前",
    "验证",
    "轮次",
    "支持集",
    "方程",
    "策略",
    "变量",
    "实验",
    "物理",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _normalize_tokens(*texts: str) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(text.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            counter[token] += 1
    return [token for token, _ in counter.most_common(40)]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_between(text: str, begin: str, end: str) -> str:
    start = text.find(begin)
    finish = text.find(end)
    if start == -1 or finish == -1 or finish <= start:
        return ""
    return text[start + len(begin):finish].strip()


@dataclass
class TaskSignature:
    task_id: str
    task_hash: str
    domain_tags: list[str]
    objective_tags: list[str]
    data_tags: list[str]
    keywords: list[str]
    preview: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class L3MemoryStore:
    """Persistent cross-task experience store for Hamilton."""

    def __init__(self, root: str | Path, top_k: int = 6, retrieval_mode: str = "hybrid"):
        self.root = Path(root)
        self.top_k = max(1, int(top_k))
        self.retrieval_mode = retrieval_mode
        self.index_file = self.root / L3_INDEX_FILE
        self.tasks_dir = self.root / L3_TASKS_DIR

    def ensure_store(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self.index_file.write_text("", encoding="utf-8")

    def build_task_signature(self, task_description: str, task_id: str = "unknown_task") -> TaskSignature:
        text = task_description.strip()
        lower = text.lower()

        domain_tags: list[str] = []
        if "vortex-induced vibration" in lower or "viv" in lower:
            domain_tags.extend(["viv", "ode_dynamics", "bridge"])
        if "newtonbench" in lower or "module:" in lower:
            domain_tags.extend(["newtonbench", "law_discovery"])
        if "kuramoto" in lower:
            domain_tags.extend(["kuramoto", "network_dynamics"])
        if "symbolic regression" in lower or "equation" in lower or "方程" in task_description:
            domain_tags.append("symbolic_regression")

        objective_tags: list[str] = []
        if "support set" in lower or "支持集" in task_description:
            objective_tags.append("support_set_recovery")
        if "redund" in lower or "冗余" in task_description or "proxy" in lower or "代理" in task_description:
            objective_tags.append("redundancy_aware")
        if "ood" in lower or "泛化" in task_description:
            objective_tags.append("ood_generalization")
        if "physics" in lower or "物理" in task_description:
            objective_tags.append("physics_consistency")
        if "trajectory" in lower or "极限环" in task_description or "solve_ivp" in lower:
            objective_tags.append("trajectory_validation")

        data_tags = _dedupe(CSV_RE.findall(task_description))
        keywords = _normalize_tokens(task_description)
        preview = text[:500]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

        return TaskSignature(
            task_id=task_id,
            task_hash=digest,
            domain_tags=_dedupe(domain_tags),
            objective_tags=_dedupe(objective_tags),
            data_tags=data_tags,
            keywords=keywords,
            preview=preview,
            created_at=_utc_now(),
        )

    def materialize_runtime_context(self, workspace: Path, signature: TaskSignature) -> dict[str, Any]:
        self.ensure_store()
        _write_json(workspace / TASK_SIGNATURE_FILE, signature.to_dict())

        hits = self.retrieve(signature)
        _write_json(workspace / L3_HITS_FILE, hits)
        (workspace / L3_CONTEXT_FILE).write_text(self._render_transfer_markdown(signature, hits), encoding="utf-8")
        (workspace / CRITIC_CONTEXT_FILE).write_text(self._render_critic_markdown(signature, hits), encoding="utf-8")

        debate_state = _read_json(workspace / DEBATE_STATE_FILE, {})
        if not isinstance(debate_state, dict):
            debate_state = {}
        debate_state.setdefault("unresolved_challenges", [])
        debate_state.setdefault("resolved_challenges", [])
        debate_state.setdefault("rounds", [])
        debate_state["task_id"] = signature.task_id
        debate_state["task_hash"] = signature.task_hash
        debate_state["updated_at"] = _utc_now()
        _write_json(workspace / DEBATE_STATE_FILE, debate_state)
        return hits

    def retrieve(self, signature: TaskSignature) -> dict[str, Any]:
        cards = _read_jsonl(self.index_file)
        scored: list[tuple[float, dict[str, Any]]] = []
        for card in cards:
            score = self._score_card(card, signature)
            if score <= 0:
                continue
            card_copy = dict(card)
            card_copy["score"] = round(score, 4)
            scored.append((score, card_copy))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_cards = [card for _, card in scored[: self.top_k]]
        return {
            "retrieval_mode": self.retrieval_mode,
            "generated_at": _utc_now(),
            "task_signature": signature.to_dict(),
            "cards": top_cards,
        }

    def promote_task(
        self,
        workspace: Path,
        task_description: str,
        task_id: str = "unknown_task",
        experiment_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_store()
        signature_payload = _read_json(workspace / TASK_SIGNATURE_FILE, {})
        if signature_payload:
            signature = TaskSignature(**signature_payload)
        else:
            signature = self.build_task_signature(task_description, task_id=task_id)

        cards = self._extract_cards(workspace, signature, task_description, experiment_record or {})
        task_dir = self.tasks_dir / signature.task_hash
        task_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "task_id": signature.task_id,
            "task_hash": signature.task_hash,
            "task_signature": signature.to_dict(),
            "card_count": len(cards),
            "promoted_at": _utc_now(),
            "retrieval_mode": self.retrieval_mode,
        }
        _write_json(task_dir / "task_signature.json", signature.to_dict())
        _write_json(task_dir / "task_summary.json", summary)
        _write_jsonl(task_dir / "cards.jsonl", cards)

        existing = _read_jsonl(self.index_file)
        existing_ids = {record.get("card_id") for record in existing}
        merged = existing + [card for card in cards if card.get("card_id") not in existing_ids]
        _write_jsonl(self.index_file, merged)

        return summary

    def _score_card(self, card: dict[str, Any], signature: TaskSignature) -> float:
        score = 0.0
        card_domains = set(card.get("domain_tags") or [])
        card_objectives = set(card.get("objective_tags") or [])
        card_data = set(card.get("data_tags") or [])
        card_keywords = set(card.get("keywords") or [])

        score += 3.0 * len(card_domains & set(signature.domain_tags))
        score += 2.5 * len(card_objectives & set(signature.objective_tags))
        score += 1.0 * len(card_data & set(signature.data_tags))
        score += 0.35 * len(card_keywords & set(signature.keywords))
        score += float(card.get("confidence", 0.5))
        score += 0.5 * float(card.get("evidence_strength", card.get("confidence", 0.5)))

        if card.get("task_hash") == signature.task_hash:
            score -= 100.0
        return score

    def _render_transfer_markdown(self, signature: TaskSignature, hits: dict[str, Any]) -> str:
        lines = [
            "# L3 跨任务经验",
            "",
            f"- task_id: {signature.task_id}",
            f"- task_hash: {signature.task_hash}",
            f"- domain_tags: {', '.join(signature.domain_tags) or '无'}",
            f"- objective_tags: {', '.join(signature.objective_tags) or '无'}",
            "",
            "## 检索命中",
        ]
        for idx, card in enumerate(hits.get("cards", []), start=1):
            lines.extend(
                [
                    f"### {idx}. [{card.get('card_type', 'unknown')}] {card.get('title', 'Untitled')}",
                    f"- polarity: {card.get('polarity', 'positive')}",
                    f"- producer_role: {card.get('producer_role', 'system')}",
                    f"- score: {card.get('score', 0)}",
                    f"- source_task: {card.get('task_id', 'unknown')} ({card.get('task_hash', 'n/a')})",
                    f"- summary: {card.get('summary', '')}",
                    f"- applicability: {card.get('applicability', '')}",
                    f"- survived_attack: {card.get('survived_attack', False)}",
                    "",
                ]
            )
        if not hits.get("cards"):
            lines.append("（暂无跨任务经验，按当前任务自主探索）")
        return "\n".join(lines).strip() + "\n"

    def _render_critic_markdown(self, signature: TaskSignature, hits: dict[str, Any]) -> str:
        lines = [
            "# Critic 审核参考",
            "",
            f"- task_id: {signature.task_id}",
            f"- challenge_taxonomy: {', '.join(CRITIC_CHALLENGE_TYPES)}",
            "",
            "## 优先审查的跨任务失败模式",
        ]
        failure_cards = [
            card
            for card in hits.get("cards", [])
            if card.get("polarity") == "negative"
            or card.get("card_type") in {"failure_card", "validation_rubric"}
        ]
        for idx, card in enumerate(failure_cards, start=1):
            lines.extend(
                [
                    f"### {idx}. [{card.get('card_type', 'unknown')}] {card.get('title', 'Untitled')}",
                    f"- summary: {card.get('summary', '')}",
                    f"- producer_role: {card.get('producer_role', 'system')}",
                    f"- negative_evidence: {card.get('negative_evidence', '')}",
                    f"- source_task: {card.get('task_id', 'unknown')}",
                    "",
                ]
            )
        if not failure_cards:
            lines.append("（暂无历史 failure card，按固定 taxonomy 审查）")
        return "\n".join(lines).strip() + "\n"

    def _extract_cards(
        self,
        workspace: Path,
        signature: TaskSignature,
        task_description: str,
        experiment_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []

        findings = _read_text(workspace / "findings.md")
        plan = _read_text(workspace / "plan.md")
        hcc_ledger = _read_jsonl(workspace / HCC_LEDGER_FILE)
        variable_memory = _read_json(workspace / "variable_memory.json", {})
        routing_state = _read_json(workspace / "routing_state.json", {})
        hypothesis_archive = _read_jsonl(workspace / "hypothesis_archive.jsonl")
        falsification_log = _read_jsonl(workspace / "falsification_log.jsonl")
        debate_state = _read_json(workspace / DEBATE_STATE_FILE, {})
        verified_positive_hcc_entries = [
            entry
            for entry in hcc_ledger
            if isinstance(entry, dict)
            and entry.get("polarity") == "positive"
            and entry.get("evidence_paths")
            and (not entry.get("attacked", False) or entry.get("survived_attack", False))
        ]
        allow_positive_transfer = bool(verified_positive_hcc_entries)

        if hcc_ledger:
            cards.extend(self._extract_cards_from_hcc_ledger(signature, hcc_ledger, allow_positive_transfer))

        current_best = self._extract_current_best(plan)
        if current_best and allow_positive_transfer:
            cards.append(
                self._build_card(
                    signature,
                    card_type="domain_prior",
                    title="当前最优结构与任务摘要",
                    summary=current_best,
                    applicability="用于下游任务的初始假设和结构先验。",
                    confidence=0.65,
                    source="plan.md",
                    polarity="positive",
                    producer_role="hamilton",
                    consumer_scope="both",
                    survived_attack=True,
                    evidence_strength=0.8,
                )
            )

        if findings and allow_positive_transfer:
            insight_lines = [line.strip("- ").strip() for line in findings.splitlines() if line.startswith("- ")]
            if insight_lines:
                cards.append(
                    self._build_card(
                        signature,
                        card_type="validation_rubric",
                        title="任务中高频出现的验证关注点",
                        summary="；".join(insight_lines[:5]),
                        applicability="用于约束跨任务验证 checklist。",
                        confidence=0.55,
                        source="findings.md",
                        polarity="positive",
                        producer_role="system",
                        consumer_scope="both",
                        survived_attack=allow_positive_transfer,
                        evidence_strength=0.55,
                    )
                )

        variables = variable_memory.get("variables", {}) if isinstance(variable_memory, dict) else {}
        if allow_positive_transfer:
            for variable_name, info in variables.items():
                if not isinstance(info, dict):
                    continue
                role = info.get("role", "unknown")
                if role in {"", "unknown"}:
                    continue
                evidence = info.get("evidence", {})
                notes = info.get("notes", [])
                cards.append(
                    self._build_card(
                        signature,
                        card_type="support_set_prior",
                        title=f"变量 {variable_name} 的角色判定",
                        summary=f"role={role}; evidence={json.dumps(evidence, ensure_ascii=False)}",
                        applicability="用于新任务中的变量角色初始化和支持集筛选。",
                        confidence=0.7 if role in {"core", "redundant", "proxy", "spurious"} else 0.55,
                        source="variable_memory.json",
                        negative_evidence="；".join(str(note) for note in notes[:3]),
                        polarity="positive",
                        producer_role="hamilton",
                        consumer_scope="both",
                        survived_attack=allow_positive_transfer,
                        evidence_strength=0.75 if role in {"core", "redundant", "proxy", "spurious"} else 0.55,
                    )
                )

        strategy_history = routing_state.get("strategy_history", []) if isinstance(routing_state, dict) else []
        if strategy_history and allow_positive_transfer:
            strategy_preview = []
            for item in strategy_history[-3:]:
                if isinstance(item, dict):
                    strategy_preview.append(
                        f"{item.get('strategy', item.get('name', 'unknown'))}: {item.get('reason', item.get('notes', ''))}"
                    )
                else:
                    strategy_preview.append(str(item))
            cards.append(
                self._build_card(
                    signature,
                    card_type="tool_recipe",
                    title="最近有效的工具路由经验",
                    summary="；".join(strategy_preview),
                    applicability="用于新任务的轮级 tool router 初始化。",
                    confidence=0.6,
                    source="routing_state.json",
                    polarity="positive",
                    producer_role="hamilton",
                    consumer_scope="hamilton",
                    survived_attack=allow_positive_transfer,
                    evidence_strength=0.6,
                )
            )

        if allow_positive_transfer:
            for item in hypothesis_archive[-5:]:
                if not isinstance(item, dict):
                    continue
                equation = item.get("equation")
                if not equation:
                    continue
                support_set = item.get("support_set", [])
                cards.append(
                    self._build_card(
                        signature,
                        card_type="operator_motif",
                        title=f"候选结构: {equation[:80]}",
                        summary=f"support_set={support_set}; tool_path={item.get('tool_path', '')}",
                        applicability="用于新任务的结构先验和 operator motif 候选。",
                        confidence=0.55,
                        source="hypothesis_archive.jsonl",
                        polarity="positive",
                        producer_role="hamilton",
                        consumer_scope="both",
                        survived_attack=allow_positive_transfer,
                        evidence_strength=0.7,
                    )
                )

        for item in falsification_log[-8:]:
            if not isinstance(item, dict):
                continue
            cards.append(
                self._build_card(
                    signature,
                    card_type="failure_card",
                    title=item.get("title") or item.get("hypothesis_id") or "历史证伪记录",
                    summary=item.get("result") or item.get("summary") or json.dumps(item, ensure_ascii=False)[:300],
                    applicability="用于 Critic 优先攻击历史上高频失效模式。",
                    confidence=0.7,
                    source="falsification_log.jsonl",
                    negative_evidence=item.get("rejection_reason", ""),
                    polarity="negative",
                    producer_role="hamilton",
                    consumer_scope="both",
                    survived_attack=False,
                    evidence_strength=0.75,
                )
            )

        unresolved = debate_state.get("unresolved_challenges", []) if isinstance(debate_state, dict) else []
        for item in unresolved[-6:]:
            if not isinstance(item, dict):
                continue
            cards.append(
                self._build_card(
                    signature,
                    card_type="failure_card",
                    title=item.get("title") or item.get("challenge_type") or "未解决 challenge",
                    summary=item.get("summary") or item.get("required_evidence") or "",
                    applicability="用于跨任务提醒高风险未解问题。",
                    confidence=0.8,
                    source=DEBATE_STATE_FILE,
                    negative_evidence=item.get("blocking_reason", ""),
                    polarity="negative",
                    producer_role="critic",
                    consumer_scope="both",
                    survived_attack=False,
                    evidence_strength=0.8,
                )
            )

        if not cards:
            cards.append(
                self._build_card(
                    signature,
                    card_type="domain_prior",
                    title="任务级摘要",
                    summary=task_description[:400],
                    applicability="默认任务摘要。",
                    confidence=0.4,
                    source="task_description",
                )
            )

        deduped: dict[str, dict[str, Any]] = {}
        for card in cards:
            deduped[card["card_id"]] = card
        return list(deduped.values())

    def _build_card(
        self,
        signature: TaskSignature,
        *,
        card_type: str,
        title: str,
        summary: str,
        applicability: str,
        confidence: float,
        source: str,
        negative_evidence: str = "",
        polarity: str = "positive",
        producer_role: str = "system",
        consumer_scope: str = "both",
        survived_attack: bool = False,
        evidence_strength: float | None = None,
    ) -> dict[str, Any]:
        keywords = _normalize_tokens(title, summary, applicability, negative_evidence)
        raw = f"{signature.task_hash}|{card_type}|{title}|{summary}|{source}"
        card_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
        return {
            "card_id": card_id,
            "task_id": signature.task_id,
            "task_hash": signature.task_hash,
            "card_type": card_type,
            "title": title[:160],
            "summary": summary[:1000],
            "applicability": applicability[:400],
            "negative_evidence": negative_evidence[:400],
            "confidence": round(max(0.0, min(confidence, 1.0)), 3),
            "polarity": "negative" if polarity == "negative" else "positive",
            "producer_role": producer_role,
            "consumer_scope": consumer_scope,
            "survived_attack": bool(survived_attack),
            "evidence_strength": round(max(0.0, min(float(evidence_strength if evidence_strength is not None else confidence), 1.0)), 3),
            "domain_tags": signature.domain_tags,
            "objective_tags": signature.objective_tags,
            "data_tags": signature.data_tags,
            "keywords": keywords,
            "source": source,
            "retrieval_mode": self.retrieval_mode,
            "created_at": _utc_now(),
        }

    def _extract_cards_from_hcc_ledger(
        self,
        signature: TaskSignature,
        ledger_entries: list[dict[str, Any]],
        allow_positive_transfer: bool,
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for entry in ledger_entries:
            if not isinstance(entry, dict):
                continue
            polarity = "negative" if entry.get("polarity") == "negative" else "positive"
            attacked = bool(entry.get("attacked", False))
            survived_attack = bool(entry.get("survived_attack", False))
            if polarity == "positive":
                if not allow_positive_transfer:
                    continue
                if not entry.get("evidence_paths"):
                    continue
                if attacked and not survived_attack:
                    continue
            card_type = entry.get("card_type") or ("failure_card" if polarity == "negative" else "domain_prior")
            cards.append(
                self._build_card(
                    signature,
                    card_type=card_type,
                    title=entry.get("title", "HCC 经验"),
                    summary=entry.get("summary", ""),
                    applicability="来自共享 HCC ledger 的跨任务迁移经验。",
                    confidence=float(entry.get("evidence_strength", entry.get("confidence", 0.6)) or 0.6),
                    source=entry.get("source", HCC_LEDGER_FILE),
                    negative_evidence=entry.get("negative_evidence", ""),
                    polarity=polarity,
                    producer_role=str(entry.get("producer_role", "system")),
                    consumer_scope=str(entry.get("consumer_scope", "both")),
                    survived_attack=survived_attack,
                    evidence_strength=float(entry.get("evidence_strength", 0.6) or 0.6),
                )
            )
        return cards

    def _extract_current_best(self, plan_text: str) -> str:
        from .constants import CURRENT_BEST_BEGIN, CURRENT_BEST_END

        block = _extract_between(plan_text, CURRENT_BEST_BEGIN, CURRENT_BEST_END)
        if block:
            return block
        if "## 当前最优" in plan_text:
            after = plan_text.split("## 当前最优", 1)[1]
            return after.split("##", 1)[0].strip()
        return ""
