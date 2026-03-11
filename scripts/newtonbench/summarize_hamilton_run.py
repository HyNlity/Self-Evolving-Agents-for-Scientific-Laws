#!/usr/bin/env python3
"""Summarize Hamilton NewtonBench batch run results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


KEY_VALUE_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
FINAL_LAW_BLOCK = re.compile(r"<final_law>\s*(.*?)\s*</final_law>", re.S)
LOG_LINE_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+.*?\s+-\s+[A-Z]+\s+-\s*"
)


@dataclass
class TaskRecord:
    task_id: str
    description: str
    module: str | None
    system: str | None
    difficulty: str | None
    law_version: str | None
    noise: float | None
    code_assisted: bool | None
    status: str
    run_experiment_calls: int
    run_experiment_success_calls: int
    evaluate_calls: int
    evaluate_success_calls: int
    has_final_law: bool
    final_law: str
    task_completed: bool | None
    l2_not_updated_warning: bool
    total_tokens: int
    exact_accuracy: float | None = None
    rmsle: float | None = None
    symbolic_equivalent: bool | None = None
    symbolic_msg: str | None = None
    evaluation_error: str | None = None

    @property
    def protocol_core_ok(self) -> bool:
        task_done = self.task_completed if self.task_completed is not None else (self.status == "completed")
        return bool(task_done) and self.run_experiment_success_calls > 0 and self.has_final_law

    @property
    def protocol_full_ok(self) -> bool:
        return self.protocol_core_ok and self.evaluate_success_calls > 0


def parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    return None


def parse_task_description(desc: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in desc.splitlines():
        m = KEY_VALUE_LINE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        out[key] = value

    if "noise" in out:
        try:
            out["noise"] = float(out["noise"])
        except Exception:
            pass
    if "code_assisted" in out and isinstance(out["code_assisted"], str):
        parsed_bool = parse_bool(out["code_assisted"])
        if parsed_bool is not None:
            out["code_assisted"] = parsed_bool

    return out


def extract_task_description_from_log(log_text: str) -> str:
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        pos = line.find("Task:")
        if pos < 0:
            continue
        first = line[pos + len("Task:") :].strip()
        collected = [first] if first else []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if re.match(r"^\d{4}-\d{2}-\d{2}\s", nxt):
                break
            collected.append(nxt.rstrip())
            j += 1
        return "\n".join(collected).strip()
    return ""


def parse_tasks(task_file: Path | None, run_dir: Path) -> list[dict[str, Any]]:
    if task_file is not None:
        raw = json.loads(task_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("task-file must be a JSON list.")
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if isinstance(item, str):
                out.append({"id": f"task_{idx}", "description": item})
            elif isinstance(item, dict):
                obj = dict(item)
                obj.setdefault("id", f"task_{idx}")
                obj.setdefault("description", "")
                out.append(obj)
            else:
                raise ValueError(f"Unsupported task entry type: {type(item).__name__}")
        return out

    logs_dir = run_dir / "logs"
    out = []
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("*.log")):
            out.append({"id": p.stem, "description": ""})
    return out


def safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_status(task_id: str, log_text: str) -> str:
    m = re.search(rf"Task\s+{re.escape(task_id)}\s+completed:\s+([A-Za-z_]+)", log_text)
    if m:
        return m.group(1).lower()
    if "✅ Agent finished task" in log_text:
        return "completed"
    if "Reached max turns limit" in log_text:
        return "failed"
    return "unknown"


def extract_final_law(log_text: str) -> str:
    matches = FINAL_LAW_BLOCK.findall(log_text)
    if matches:
        return matches[-1].strip()
    return ""


def strip_log_prefix(line: str) -> str:
    return LOG_LINE_PREFIX.sub("", line)


def extract_finish_from_log(log_text: str) -> tuple[str, bool | None]:
    lines = log_text.splitlines()
    last_message = ""
    last_task_completed: bool | None = None

    i = 0
    while i < len(lines):
        if "📝 Finish Tool Arguments:" not in lines[i]:
            i += 1
            continue

        message_lines: list[str] = []
        task_completed: bool | None = None
        i += 1
        while i < len(lines):
            stripped = strip_log_prefix(lines[i])
            if stripped.startswith("task_completed:"):
                raw_value = stripped.split(":", 1)[1].strip()
                task_completed = parse_bool(raw_value)
                break
            if stripped.startswith("message:"):
                message_lines.append(stripped[len("message:") :].lstrip())
            elif message_lines:
                if stripped == "=" * 80:
                    break
                message_lines.append(stripped)
            i += 1

        if message_lines:
            last_message = "\n".join(message_lines).strip()
        if task_completed is not None:
            last_task_completed = task_completed
        i += 1

    return last_message, last_task_completed


def extract_experiment_record_path_from_log(log_text: str, repo_root: Path) -> Path | None:
    """Extract experiment record path from task log."""
    record_path_text: str | None = None
    for line in log_text.splitlines():
        stripped = strip_log_prefix(line)
        marker = "Experiment record saved to "
        if marker not in stripped:
            continue
        record_path_text = stripped.split(marker, 1)[1].strip()

    if not record_path_text:
        return None

    candidate = Path(record_path_text)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    if candidate.exists():
        return candidate
    return None


def parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_bool(value)
    return None


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return None


def parse_round_rows_from_experiment_record(record_path: Path) -> list[dict[str, Any]]:
    """Parse per-round protocol/evaluation rows from experiment record JSON."""
    if not record_path.exists():
        return []
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in rounds:
        if not isinstance(item, dict):
            continue
        signal = item.get("signal", {})
        if not isinstance(signal, dict):
            signal = {}
        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
        last_eval = protocol.get("last_evaluation", {})
        if not isinstance(last_eval, dict):
            last_eval = {}
        violations = protocol.get("violations", [])
        if not isinstance(violations, list):
            violations = []

        rows.append(
            {
                "round": item.get("round"),
                "satisfied": bool(signal.get("satisfied", False)),
                "task_completed": parse_optional_bool(signal.get("task_completed")),
                "run_experiment_calls": int(protocol.get("run_experiment_calls", 0) or 0),
                "run_experiment_success_calls": int(protocol.get("run_experiment_success_calls", 0) or 0),
                "evaluate_calls": int(protocol.get("evaluate_submission_calls", 0) or 0),
                "evaluate_success_calls": int(protocol.get("evaluate_submission_success_calls", 0) or 0),
                "has_final_law_block": bool(protocol.get("has_final_law_block", False)),
                "signature_match": parse_optional_bool(protocol.get("signature_match")),
                "rmsle": safe_float(last_eval.get("rmsle")),
                "exact_accuracy": safe_float(last_eval.get("exact_accuracy")),
                "symbolic_equivalent": parse_optional_bool(last_eval.get("symbolic_equivalent")),
                "violations": [str(v) for v in violations],
            }
        )
    return rows


def summarize_round_rows(round_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-round rows for one task."""
    rmsle_values = [
        float(row["rmsle"])
        for row in round_rows
        if row.get("rmsle") is not None and math.isfinite(float(row.get("rmsle")))
    ]
    last_rmsle = safe_float(round_rows[-1].get("rmsle")) if round_rows else None
    rmsle_path_parts: list[str] = []
    for row in round_rows:
        round_id = row.get("round")
        round_rmsle = safe_float(row.get("rmsle"))
        label = "NA" if round_rmsle is None else f"{round_rmsle:.6g}"
        rmsle_path_parts.append(f"R{round_id}:{label}")

    return {
        "rounds_total": len(round_rows),
        "rounds_satisfied_true": sum(1 for row in round_rows if row.get("satisfied") is True),
        "rounds_task_completed_true": sum(1 for row in round_rows if row.get("task_completed") is True),
        "rounds_task_completed_false": sum(1 for row in round_rows if row.get("task_completed") is False),
        "round_run_experiment_calls_total": sum(
            int(row.get("run_experiment_calls", 0) or 0) for row in round_rows
        ),
        "round_run_experiment_success_calls_total": sum(
            int(row.get("run_experiment_success_calls", 0) or 0) for row in round_rows
        ),
        "round_evaluate_calls_total": sum(
            int(row.get("evaluate_calls", 0) or 0) for row in round_rows
        ),
        "round_evaluate_success_calls_total": sum(
            int(row.get("evaluate_success_calls", 0) or 0) for row in round_rows
        ),
        "rounds_with_eval_success": sum(1 for row in round_rows if int(row.get("evaluate_success_calls", 0) or 0) > 0),
        "rounds_with_final_law_block": sum(1 for row in round_rows if row.get("has_final_law_block") is True),
        "round_best_rmsle": min(rmsle_values) if rmsle_values else None,
        "round_last_rmsle": last_rmsle,
        "round_rmsle_path": " | ".join(rmsle_path_parts),
    }


def build_rounds_summary(task_rounds_payload: list[dict[str, Any]]) -> dict[str, Any]:
    """Build run-level summary over all round rows."""
    all_rows: list[dict[str, Any]] = []
    for task_payload in task_rounds_payload:
        rows = task_payload.get("rounds", [])
        if isinstance(rows, list):
            all_rows.extend([r for r in rows if isinstance(r, dict)])

    rmsle_values = [
        float(row["rmsle"])
        for row in all_rows
        if row.get("rmsle") is not None and math.isfinite(float(row.get("rmsle")))
    ]

    return {
        "tasks_with_round_data": len(task_rounds_payload),
        "total_rounds": len(all_rows),
        "satisfied_rounds": sum(1 for row in all_rows if row.get("satisfied") is True),
        "task_completed_true_rounds": sum(1 for row in all_rows if row.get("task_completed") is True),
        "task_completed_false_rounds": sum(1 for row in all_rows if row.get("task_completed") is False),
        "run_experiment_calls_total": sum(int(row.get("run_experiment_calls", 0) or 0) for row in all_rows),
        "run_experiment_success_calls_total": sum(
            int(row.get("run_experiment_success_calls", 0) or 0) for row in all_rows
        ),
        "evaluate_calls_total": sum(int(row.get("evaluate_calls", 0) or 0) for row in all_rows),
        "evaluate_success_calls_total": sum(
            int(row.get("evaluate_success_calls", 0) or 0) for row in all_rows
        ),
        "rounds_with_eval_success": sum(
            1 for row in all_rows if int(row.get("evaluate_success_calls", 0) or 0) > 0
        ),
        "avg_round_rmsle": mean(rmsle_values) if rmsle_values else None,
        "best_round_rmsle": min(rmsle_values) if rmsle_values else None,
    }


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Extract the last JSON object from mixed stdout text."""
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    last_obj: dict[str, Any] | None = None

    while idx < n:
        ch = text[idx]
        if ch != "{":
            idx += 1
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(value, dict):
            last_obj = value
        idx = end

    return last_obj


def safe_json_loads_dict(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def collect_script_stats_from_trajectory(
    trajectory_path: Path,
) -> tuple[dict[str, dict[str, int]], str, bool | None, list[dict[str, Any]]]:
    """Extract per-script call/success stats from assistant tool_calls in trajectory."""
    if not trajectory_path.exists():
        return {}, "", None, []
    try:
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "", None, []

    if not isinstance(data, list) or not data:
        return {}, "", None, []
    last = data[-1]
    if not isinstance(last, dict):
        return {}, "", None, []
    trajectory = last.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return {}, "", None, []
    dialogs = trajectory.get("dialogs", [])
    if not isinstance(dialogs, list):
        return {}, "", None, []

    messages: list[dict[str, Any]] = []
    for dialog in dialogs:
        if not isinstance(dialog, dict):
            continue
        dialog_messages = dialog.get("messages", [])
        if isinstance(dialog_messages, list):
            for msg in dialog_messages:
                if isinstance(msg, dict):
                    messages.append(msg)

    tool_payload_by_id: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if not isinstance(tcid, str) or not tcid:
            continue
        content = msg.get("content", "")
        meta = msg.get("meta", {})
        info = meta.get("info", {}) if isinstance(meta, dict) else {}
        tool_payload_by_id[tcid] = {
            "info": info if isinstance(info, dict) else {},
            "content": content if isinstance(content, str) else str(content),
        }

    stats: dict[str, dict[str, int]] = {}
    finish_message = ""
    finish_task_completed: bool | None = None
    script_records: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if not isinstance(fn, dict):
                continue
            fn_name = fn.get("name")
            if not isinstance(fn_name, str):
                continue

            args = safe_json_loads_dict(fn.get("arguments", "") or "")
            if fn_name == "finish":
                if isinstance(args, dict):
                    msg_text = args.get("message")
                    if isinstance(msg_text, str):
                        finish_message = msg_text
                    task_completed_value = args.get("task_completed")
                    if isinstance(task_completed_value, bool):
                        finish_task_completed = task_completed_value
                    elif isinstance(task_completed_value, str):
                        finish_task_completed = parse_bool(task_completed_value)
                continue

            if fn_name != "use_skill":
                continue
            if not isinstance(args, dict):
                continue
            if args.get("action") != "run_script":
                continue

            script_name = str(args.get("script_name", "") or "").strip()
            if not script_name:
                continue

            rec = stats.setdefault(script_name, {"calls": 0, "success_calls": 0})
            rec["calls"] += 1

            tcid = tc.get("id")
            payload = tool_payload_by_id.get(tcid, {}) if isinstance(tcid, str) else {}
            info = payload.get("info", {}) if isinstance(payload, dict) else {}
            output = payload.get("content", "") if isinstance(payload, dict) else ""
            exit_code = info.get("exit_code") if isinstance(info, dict) else None
            try:
                ok = exit_code is None or int(exit_code) == 0
            except Exception:
                ok = str(exit_code).strip() == "0"
            if ok:
                rec["success_calls"] += 1
            script_records.append(
                {
                    "script_name": script_name,
                    "success": ok,
                    "output": output if isinstance(output, str) else str(output),
                }
            )

    return stats, finish_message, finish_task_completed, script_records


def parse_total_tokens(trajectory_path: Path) -> int:
    if not trajectory_path.exists():
        return 0
    try:
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    total = 0
    trajectory = data[-1].get("trajectory", {}) if isinstance(data, list) and data else {}
    dialogs = trajectory.get("dialogs", [])
    if not isinstance(dialogs, list):
        return 0

    for dialog in dialogs:
        messages = dialog.get("messages", [])
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            meta = msg.get("meta", {})
            usage = meta.get("usage", {}) if isinstance(meta, dict) else {}
            if isinstance(usage, dict):
                try:
                    total += int(usage.get("total_tokens", 0) or 0)
                except Exception:
                    pass
    return total


def extract_last_logged_evaluation(log_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    idx = 0
    n = len(log_text)
    last_eval: dict[str, Any] | None = None

    while idx < n:
        if log_text[idx] != "{":
            idx += 1
            continue
        try:
            value, end = decoder.raw_decode(log_text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(value, dict):
            evaluation = value.get("evaluation")
            if isinstance(evaluation, dict):
                last_eval = evaluation
        idx = end

    return last_eval


def extract_last_trajectory_evaluation(script_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    last_eval: dict[str, Any] | None = None
    for rec in script_records:
        if rec.get("script_name") != "evaluate_submission.py":
            continue
        if not rec.get("success"):
            continue
        payload = extract_last_json_object(str(rec.get("output", "") or ""))
        if not isinstance(payload, dict):
            continue
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, dict):
            last_eval = evaluation
    return last_eval


def apply_evaluation_to_record(record: TaskRecord, evaluation: dict[str, Any]) -> None:
    runtime_error = evaluation.get("error")
    if isinstance(runtime_error, str) and runtime_error.strip():
        record.evaluation_error = runtime_error.strip()

    try:
        if "exact_accuracy" in evaluation:
            value = float(evaluation.get("exact_accuracy"))
            if math.isfinite(value):
                record.exact_accuracy = value
    except Exception:
        pass
    try:
        if "rmsle" in evaluation and evaluation.get("rmsle") is not None:
            value = float(evaluation.get("rmsle"))
            if math.isfinite(value):
                record.rmsle = value
            elif record.evaluation_error is None:
                record.evaluation_error = "non_finite_rmsle"
    except Exception:
        pass

    symbolic_equivalent = evaluation.get("symbolic_equivalent")
    if isinstance(symbolic_equivalent, bool):
        record.symbolic_equivalent = symbolic_equivalent
    symbolic_msg = evaluation.get("symbolic_msg")
    if isinstance(symbolic_msg, str):
        record.symbolic_msg = symbolic_msg


def run_auto_evaluation(
    repo_root: Path,
    record: TaskRecord,
    judge_model: str,
    newtonbench_root: str | None,
) -> None:
    if not record.final_law.strip():
        record.evaluation_error = "final_law_empty"
        return
    if not record.module or not record.difficulty:
        record.evaluation_error = "missing_task_metadata"
        return

    eval_script = repo_root / "evomaster/skills/newtonbench/scripts/evaluate_submission.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--module",
        record.module,
        "--difficulty",
        record.difficulty,
        "--law-version",
        record.law_version or "v0",
        "--judge-model",
        judge_model,
        "--law-text",
        record.final_law,
    ]
    if newtonbench_root:
        cmd.extend(["--newtonbench-root", newtonbench_root])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        record.evaluation_error = stderr or f"evaluate_submission_exit_{proc.returncode}"
        return

    stdout = proc.stdout.strip()
    payload = extract_last_json_object(stdout)
    if payload is None:
        record.evaluation_error = "invalid_evaluation_json"
        return

    evaluation = payload.get("evaluation", {})
    if not isinstance(evaluation, dict):
        record.evaluation_error = "evaluation_missing"
        return

    apply_evaluation_to_record(record, evaluation)


def build_summary(records: list[TaskRecord]) -> dict[str, Any]:
    total = len(records)
    completed = sum(1 for r in records if r.status == "completed")
    task_completed_true = sum(1 for r in records if r.task_completed is True)
    task_completed_false = sum(1 for r in records if r.task_completed is False)
    with_final_law = sum(1 for r in records if r.has_final_law)
    with_experiment = sum(1 for r in records if r.run_experiment_calls > 0)
    with_experiment_success = sum(1 for r in records if r.run_experiment_success_calls > 0)
    with_eval_call = sum(1 for r in records if r.evaluate_calls > 0)
    with_eval_success = sum(1 for r in records if r.evaluate_success_calls > 0)
    protocol_ok = sum(1 for r in records if r.protocol_core_ok)
    protocol_full_ok = sum(1 for r in records if r.protocol_full_ok)
    with_eval = sum(1 for r in records if r.exact_accuracy is not None or r.rmsle is not None)
    symbolic_true = sum(1 for r in records if r.symbolic_equivalent is True)

    exact_list = [
        r.exact_accuracy
        for r in records
        if r.exact_accuracy is not None and math.isfinite(r.exact_accuracy)
    ]
    rmsle_list = [
        r.rmsle
        for r in records
        if r.rmsle is not None and math.isfinite(r.rmsle)
    ]
    token_list = [r.total_tokens for r in records if r.total_tokens > 0]

    return {
        "total_tasks": total,
        "completed_tasks": completed,
        "task_completed_true": task_completed_true,
        "task_completed_false": task_completed_false,
        "with_final_law": with_final_law,
        "with_run_experiment": with_experiment,
        "with_run_experiment_success": with_experiment_success,
        "with_evaluate_call": with_eval_call,
        "with_evaluate_success": with_eval_success,
        "total_run_experiment_calls": sum(int(r.run_experiment_calls or 0) for r in records),
        "total_run_experiment_success_calls": sum(
            int(r.run_experiment_success_calls or 0) for r in records
        ),
        "total_evaluate_calls": sum(int(r.evaluate_calls or 0) for r in records),
        "total_evaluate_success_calls": sum(int(r.evaluate_success_calls or 0) for r in records),
        "protocol_core_ok": protocol_ok,
        "protocol_full_ok": protocol_full_ok,
        "auto_evaluated_tasks": with_eval,
        "symbolic_equivalent_true": symbolic_true,
        "exact_accuracy_tasks": len(exact_list),
        "rmsle_tasks": len(rmsle_list),
        "avg_exact_accuracy": mean(exact_list) if exact_list else None,
        "avg_rmsle": mean(rmsle_list) if rmsle_list else None,
        "avg_total_tokens": mean(token_list) if token_list else None,
    }


def to_csv_rows(
    records: list[TaskRecord],
    round_summary_by_task: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    round_summary_by_task = round_summary_by_task or {}
    for r in records:
        round_summary = round_summary_by_task.get(r.task_id, {})
        rows.append(
            {
                "task_id": r.task_id,
                "module": r.module,
                "system": r.system,
                "difficulty": r.difficulty,
                "law_version": r.law_version,
                "noise": r.noise,
                "code_assisted": r.code_assisted,
                "status": r.status,
                "task_completed": r.task_completed,
                "run_experiment_calls": r.run_experiment_calls,
                "run_experiment_success_calls": r.run_experiment_success_calls,
                "evaluate_calls": r.evaluate_calls,
                "evaluate_success_calls": r.evaluate_success_calls,
                "has_final_law": r.has_final_law,
                "protocol_core_ok": r.protocol_core_ok,
                "protocol_full_ok": r.protocol_full_ok,
                "l2_not_updated_warning": r.l2_not_updated_warning,
                "total_tokens": r.total_tokens,
                "exact_accuracy": r.exact_accuracy,
                "rmsle": r.rmsle,
                "symbolic_equivalent": r.symbolic_equivalent,
                "evaluation_error": r.evaluation_error,
                "rounds_total": round_summary.get("rounds_total"),
                "rounds_task_completed_true": round_summary.get("rounds_task_completed_true"),
                "rounds_task_completed_false": round_summary.get("rounds_task_completed_false"),
                "round_run_experiment_calls_total": round_summary.get("round_run_experiment_calls_total"),
                "round_run_experiment_success_calls_total": round_summary.get(
                    "round_run_experiment_success_calls_total"
                ),
                "round_evaluate_calls_total": round_summary.get("round_evaluate_calls_total"),
                "round_evaluate_success_calls_total": round_summary.get(
                    "round_evaluate_success_calls_total"
                ),
                "rounds_with_eval_success": round_summary.get("rounds_with_eval_success"),
                "round_best_rmsle": round_summary.get("round_best_rmsle"),
                "round_last_rmsle": round_summary.get("round_last_rmsle"),
                "round_rmsle_path": round_summary.get("round_rmsle_path"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "module",
        "system",
        "difficulty",
        "law_version",
        "noise",
        "code_assisted",
        "status",
        "task_completed",
        "run_experiment_calls",
        "run_experiment_success_calls",
        "evaluate_calls",
        "evaluate_success_calls",
        "has_final_law",
        "protocol_core_ok",
        "protocol_full_ok",
        "l2_not_updated_warning",
        "total_tokens",
        "exact_accuracy",
        "rmsle",
        "symbolic_equivalent",
        "evaluation_error",
        "rounds_total",
        "rounds_task_completed_true",
        "rounds_task_completed_false",
        "round_run_experiment_calls_total",
        "round_run_experiment_success_calls_total",
        "round_evaluate_calls_total",
        "round_evaluate_success_calls_total",
        "rounds_with_eval_success",
        "round_best_rmsle",
        "round_last_rmsle",
        "round_rmsle_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Hamilton NewtonBench run directory")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/hamilton_20260309_xxx")
    parser.add_argument("--task-file", default=None, help="Task JSON file used by run.py")
    parser.add_argument("--output-json", default=None, help="Summary JSON output path")
    parser.add_argument("--output-csv", default=None, help="Per-task CSV output path")
    parser.add_argument("--auto-evaluate", action="store_true", help="Auto-call evaluate_submission.py")
    parser.add_argument("--judge-model", default="gpt5mini", help="Judge model for auto evaluation")
    parser.add_argument("--newtonbench-root", default=None, help="Optional NewtonBench root path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    task_file = Path(args.task_file).resolve() if args.task_file else None

    tasks = parse_tasks(task_file, run_dir)
    if not tasks:
        raise ValueError("No tasks found. Provide --task-file or ensure run_dir/logs/*.log exists.")
    single_task_mode = len(tasks) == 1

    records: list[TaskRecord] = []
    round_summary_by_task: dict[str, dict[str, Any]] = {}
    task_rounds_payload: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id", ""))
        description = str(task.get("description", ""))

        log_path = run_dir / "logs" / f"{task_id}.log"
        log_text = safe_read(log_path)
        if not description.strip():
            description = extract_task_description_from_log(log_text)

        parsed = parse_task_description(description)

        status = parse_status(task_id, log_text)
        trajectory_path = run_dir / "trajectories" / task_id / "trajectory.json"
        (
            script_stats,
            finish_message,
            task_completed_from_traj,
            script_records,
        ) = collect_script_stats_from_trajectory(trajectory_path)
        run_stats = script_stats.get("run_experiment.py", {"calls": 0, "success_calls": 0})
        eval_stats = script_stats.get("evaluate_submission.py", {"calls": 0, "success_calls": 0})
        run_calls = int(run_stats.get("calls", 0))
        eval_calls = int(eval_stats.get("calls", 0))
        run_success_calls = int(run_stats.get("success_calls", 0))
        eval_success_calls = int(eval_stats.get("success_calls", 0))
        finish_message_from_log, task_completed_from_log = extract_finish_from_log(log_text)
        final_law = extract_final_law(finish_message) or extract_final_law(finish_message_from_log)
        task_completed = (
            task_completed_from_traj
            if task_completed_from_traj is not None
            else task_completed_from_log
        )
        tokens = parse_total_tokens(trajectory_path)

        record = TaskRecord(
            task_id=task_id,
            description=description,
            module=parsed.get("module"),
            system=parsed.get("system"),
            difficulty=parsed.get("difficulty"),
            law_version=parsed.get("law_version"),
            noise=parsed.get("noise") if isinstance(parsed.get("noise"), float) else None,
            code_assisted=parsed.get("code_assisted")
            if isinstance(parsed.get("code_assisted"), bool)
            else None,
            status=status,
            task_completed=task_completed,
            run_experiment_calls=run_calls,
            run_experiment_success_calls=run_success_calls,
            evaluate_calls=eval_calls,
            evaluate_success_calls=eval_success_calls,
            has_final_law=bool(final_law.strip()),
            final_law=final_law,
            l2_not_updated_warning=("L2 files not updated" in log_text),
            total_tokens=tokens,
        )

        trajectory_eval = extract_last_trajectory_evaluation(script_records)
        if trajectory_eval is not None:
            apply_evaluation_to_record(record, trajectory_eval)
        elif single_task_mode:
            logged_eval = extract_last_logged_evaluation(log_text)
            if logged_eval is not None:
                apply_evaluation_to_record(record, logged_eval)

        if args.auto_evaluate:
            run_auto_evaluation(
                repo_root=repo_root,
                record=record,
                judge_model=args.judge_model,
                newtonbench_root=args.newtonbench_root,
            )

        experiment_record_path = extract_experiment_record_path_from_log(log_text, repo_root)
        if experiment_record_path is not None:
            round_rows = parse_round_rows_from_experiment_record(experiment_record_path)
            round_summary = summarize_round_rows(round_rows)
            round_summary_by_task[task_id] = round_summary
            # Prefer cumulative per-round protocol counts over trajectory-last-round counts.
            if int(round_summary.get("rounds_total", 0) or 0) > 0:
                record.run_experiment_calls = int(round_summary.get("round_run_experiment_calls_total", 0) or 0)
                record.run_experiment_success_calls = int(
                    round_summary.get("round_run_experiment_success_calls_total", 0) or 0
                )
                record.evaluate_calls = int(round_summary.get("round_evaluate_calls_total", 0) or 0)
                record.evaluate_success_calls = int(
                    round_summary.get("round_evaluate_success_calls_total", 0) or 0
                )
            task_rounds_payload.append(
                {
                    "task_id": task_id,
                    "experiment_record_path": str(experiment_record_path),
                    "rounds_summary": round_summary,
                    "rounds": round_rows,
                }
            )
        else:
            round_summary_by_task[task_id] = summarize_round_rows([])

        records.append(record)

    summary = build_summary(records)
    rounds_summary = build_rounds_summary(task_rounds_payload)
    rows = to_csv_rows(records, round_summary_by_task=round_summary_by_task)

    output_json = Path(args.output_json) if args.output_json else run_dir / "newtonbench_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "newtonbench_trials.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "summary": summary,
                "rounds_summary": rounds_summary,
                "tasks": rows,
                "task_rounds": task_rounds_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_csv, rows)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "summary_json": str(output_json),
                "summary_csv": str(output_csv),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
