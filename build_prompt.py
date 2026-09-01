#!/usr/bin/env python3

from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


import logging
import time


# -----------------------------
# Throughput-oriented constants
# -----------------------------

# Fixed on purpose: not exposed as tuning levers.
# Keep this close to real workload needs so vLLM does not reserve KV cache
# for the model's enormous max context length.
MAX_MODEL_LEN = 40_960

# Fixed on purpose: useful default, but not a sweep lever in this script.
GPU_MEMORY_UTILIZATION = 0.95

# Fixed on purpose: deterministic structured annotation.
TEMPERATURE = 0.0
TOP_P = 1.0

# Fixed on purpose: prefix caching is useful for repeated prompt template
# prefixes and vLLM logs its hit rate every 5s.
ENABLE_PREFIX_CACHING = True

# Fixed on purpose: leave vLLM periodic stats logging enabled.
# vLLM's INFO log every ~5s includes:
# - running / waiting requests
# - GPU KV cache usage
# - prompt throughput (tok/s over recent window)
# - generation throughput (tok/s over recent window)
# - prefix cache hit rate
#
# These are the main signals to compare TP / MNS / MNBT choices.
DISABLE_LOG_STATS = False

# Fixed on purpose: throughput-first starting point for this model on H200.
# The three exposed levers are:
# - tensor_parallel_size (tp)
# - max_num_seqs (mns)
# - max_num_batched_tokens (mnbt)
DEFAULT_TP = 2
DEFAULT_MNS = 128
DEFAULT_MNBT = 65_536

# Keep your existing default if you want, but it's no longer a tuning knob.
DEFAULT_MAX_COMPLETION_TOKENS = 12_000

# Pipeline robustness knob.
DEFAULT_MAX_FILE_RETRIES = 2


DOMAIN_LABELS = [
    "SECURITY_FORENSICS",
    "DEVOPS_NETWORK_SERVICES",
    "DATA_ENGINEERING_ETL",
    "ML_DEEP_LEARNING",
    "SYSTEMS_EMULATION_VIRTUALIZATION",
    "COMPILERS_PL_FORMAL_METHODS",
    "SCIENTIFIC_COMPUTING_MATH",
    "BIOINFORMATICS_COMP_BIO",
    "DOCUMENT_IMAGE_VIDEO_PROCESSING",
    "TEXT_REGEX_EDITING",
    "GAMES_SIMULATION_MISC_CRAFT",
    "WEB_FRONTEND_UI",
    "BACKEND_API_SERVICES",
    "DATABASE_SQL_ORM",
    "TESTING_QA_TOOLING",
    "CLI_DEVELOPER_TOOLING",
    "MOBILE_EMBEDDED",
    "GPU_PARALLEL_COMPUTE",
    "NONE",
]

ATTRIBUTE_LABELS = [
    "TESTS_INCLUDED",
    "REPO_ENV_SETUP_IN_REPO",
    "METRIC_IMPROVEMENT_CLAIMED",
    "DOCS_ONLY",
    "VERSION_BUMP_ONLY",
    "API_SURFACE_CHANGE",
    "OBSERVABILITY_CHANGE",
    "SECURITY_FIX",
    "STYLE_ONLY",
    "REFACTOR_ONLY",
    "FEATURE_ADD",
    "BUGFIX",
    "GENERATED_ONLY",
]

COMPLEXITY_LABELS = [
    "KNOWLEDGE_HEAVY",
    "LOGIC_HEAVY",
    "OBSERVATION_HEAVY",
    "NONE",
]

ANNOTATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": DOMAIN_LABELS},
                "scores": {
                    "type": "object",
                    "properties": {
                        label: {"type": "integer"} for label in DOMAIN_LABELS
                    },
                    "required": DOMAIN_LABELS,
                    "additionalProperties": False,
                },
                "rationale": {"type": "string"},
            },
            "required": ["label", "scores", "rationale"],
            "additionalProperties": False,
        },
        "attributes": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": ATTRIBUTE_LABELS},
                },
                "scores": {
                    "type": "object",
                    "properties": {
                        label: {"type": "integer"} for label in ATTRIBUTE_LABELS
                    },
                    "required": ATTRIBUTE_LABELS,
                    "additionalProperties": False,
                },
                "rationale": {"type": "string"},
            },
            "required": ["labels", "scores", "rationale"],
            "additionalProperties": False,
        },
        "complexity": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": COMPLEXITY_LABELS},
                "scores": {
                    "type": "object",
                    "properties": {
                        label: {"type": "integer"} for label in COMPLEXITY_LABELS
                    },
                    "required": COMPLEXITY_LABELS,
                    "additionalProperties": False,
                },
                "rationale": {"type": "string"},
            },
            "required": ["label", "scores", "rationale"],
            "additionalProperties": False,
        },
    },
    "required": ["domain", "attributes", "complexity"],
    "additionalProperties": False,
}

MAX_TOUCHED_FILES = 200
MAX_REPO_FILES_AT_BASE = 200
MAX_COMPARE_DIFF_CHARS = 20000

MAX_COMMITS = 20
MAX_COMMIT_SUBJECT_CHARS = 300
MAX_COMMIT_PATCH_CHARS = 4000

MAX_BASE_FILES = 20
MAX_BASE_FILE_CONTENT_CHARS = 3000
MAX_BASE_FILE_PATH_CHARS = 400

MAX_EVENTS = 20
MAX_COMMENT_BODY_CHARS = 800
MAX_REVIEW_BODY_CHARS = 800
MAX_STATUS_DESC_CHARS = 300
MAX_TITLE_CHARS = 500
MAX_BODY_CHARS = 4000
MAX_LABELS = 30
MAX_PARTICIPANTS = 50
MAX_HEAD_REFS = 20
MAX_HEAD_SHAS = 20
MAX_PR_BODY_VERSIONS = 5
MAX_PR_BODY_VERSION_CHARS = 1500

MAX_PROMPT_CHARS_FALLBACK = 120000
DEFAULT_BATCH_SIZE = 32


def _get_nested(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
    return cur


def _normalize_list_field(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "list" in value:
        out = []
        for item in value["list"] or []:
            if isinstance(item, dict) and "element" in item:
                out.append(item["element"])
            else:
                out.append(item)
        return out
    return [value]


def _clean_none(x: Any) -> Any:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            cleaned = _clean_none(v)
            if cleaned is not None:
                out[k] = cleaned
        return out
    if isinstance(x, list):
        return [_clean_none(v) for v in x]
    return x


def _json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.decode("utf-8", errors="replace")
    if isinstance(x, dict):
        return {
            str(_json_safe(k)): _json_safe(v) for k, v in x.items() if v is not None
        }
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    return str(x)


def _truncate_text(x: Any, max_chars: int) -> Any:
    if x is None:
        return None
    if not isinstance(x, str):
        return x
    if len(x) <= max_chars:
        return x
    return x[:max_chars] + f"\n...[truncated {len(x) - max_chars} chars]"


def _truncate_list(items: List[Any], max_items: int) -> List[Any]:
    if len(items) <= max_items:
        return items
    kept = items[:max_items]
    kept.append(f"...[truncated {len(items) - max_items} items]")
    return kept


def batched(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def row_has_git_error(row: Dict[str, Any]) -> bool:
    return _get_nested(row, "code.debug.git_error") is not None


def build_pr_example_json(row: Dict[str, Any]) -> Dict[str, Any]:
    pull_request = row.get("pull_request") or {}
    code = row.get("code") or {}

    base_repo = _get_nested(pull_request, "base.repo", {}) or {}
    head_repo = _get_nested(pull_request, "head.repo", {}) or {}

    labels = []
    for item in _normalize_list_field(pull_request.get("labels"))[:MAX_LABELS]:
        if isinstance(item, dict):
            labels.append(
                _clean_none(
                    {
                        "name": _truncate_text(item.get("name"), 200),
                        "description": _truncate_text(item.get("description"), 300),
                        "color": item.get("color"),
                        "default": item.get("default"),
                    }
                )
            )

    events = []
    for item in _normalize_list_field(row.get("events"))[:MAX_EVENTS]:
        if isinstance(item, dict):
            events.append(
                _clean_none(
                    {
                        "id": item.get("id"),
                        "type": item.get("type"),
                        "created_at": item.get("created_at"),
                        "action": item.get("action"),
                        "comment": _clean_none(
                            {
                                "body": _truncate_text(
                                    _get_nested(item, "comment.body"),
                                    MAX_COMMENT_BODY_CHARS,
                                ),
                                "created_at": _get_nested(item, "comment.created_at"),
                                "author_association": _get_nested(
                                    item, "comment.author_association"
                                ),
                            }
                        ),
                        "review": _clean_none(
                            {
                                "state": _get_nested(item, "review.state"),
                                "body": _truncate_text(
                                    _get_nested(item, "review.body"),
                                    MAX_REVIEW_BODY_CHARS,
                                ),
                                "submitted_at": _get_nested(
                                    item, "review.submitted_at"
                                ),
                            }
                        ),
                        "status": _clean_none(
                            {
                                "state": _get_nested(item, "status.state"),
                                "context": _get_nested(item, "status.context"),
                                "description": _truncate_text(
                                    _get_nested(item, "status.description"),
                                    MAX_STATUS_DESC_CHARS,
                                ),
                            }
                        ),
                        "check_run": _clean_none(
                            {
                                "name": _get_nested(item, "check_run.name"),
                                "status": _get_nested(item, "check_run.status"),
                                "conclusion": _get_nested(item, "check_run.conclusion"),
                            }
                        ),
                        "check_suite": _clean_none(
                            {
                                "status": _get_nested(item, "check_suite.status"),
                                "conclusion": _get_nested(
                                    item, "check_suite.conclusion"
                                ),
                            }
                        ),
                    }
                )
            )

    code_commits = []
    for item in _normalize_list_field(code.get("commits"))[:MAX_COMMITS]:
        if isinstance(item, dict):
            code_commits.append(
                _clean_none(
                    {
                        "sha": item.get("sha"),
                        "subject": _truncate_text(
                            item.get("subject"), MAX_COMMIT_SUBJECT_CHARS
                        ),
                        "patch": _truncate_text(
                            item.get("patch"), MAX_COMMIT_PATCH_CHARS
                        ),
                        "patch_truncated": item.get("patch_truncated"),
                    }
                )
            )

    base_files_at_base = []
    for item in _normalize_list_field(code.get("base_files_at_base"))[:MAX_BASE_FILES]:
        if isinstance(item, dict):
            base_files_at_base.append(
                _clean_none(
                    {
                        "path": _truncate_text(
                            item.get("path"), MAX_BASE_FILE_PATH_CHARS
                        ),
                        "blob_sha": item.get("blob_sha"),
                        "size": item.get("size"),
                        "content": _truncate_text(
                            item.get("content"), MAX_BASE_FILE_CONTENT_CHARS
                        ),
                        "content_truncated": item.get("content_truncated"),
                        "error": item.get("error"),
                    }
                )
            )

    pr_body_versions = []
    for item in _normalize_list_field(row.get("pr_body_versions"))[
        :MAX_PR_BODY_VERSIONS
    ]:
        if isinstance(item, dict):
            pr_body_versions.append(
                _clean_none(
                    {
                        "source": item.get("source"),
                        "at": item.get("at"),
                        "event_id": item.get("event_id"),
                        "body": _truncate_text(
                            item.get("body"), MAX_PR_BODY_VERSION_CHARS
                        ),
                    }
                )
            )

    pr_json = {
        "repo_id": row.get("repo_id"),
        "pr_number": row.get("pr_number"),
        "pull_request": {
            "id": pull_request.get("id"),
            "number": pull_request.get("number"),
            "state": pull_request.get("state"),
            "title": _truncate_text(pull_request.get("title"), MAX_TITLE_CHARS),
            "body": _truncate_text(pull_request.get("body"), MAX_BODY_CHARS),
            "draft": pull_request.get("draft"),
            "locked": pull_request.get("locked"),
            "created_at": pull_request.get("created_at"),
            "updated_at": pull_request.get("updated_at"),
            "closed_at": pull_request.get("closed_at"),
            "merged_at": pull_request.get("merged_at"),
            "merged": pull_request.get("merged"),
            "mergeable": pull_request.get("mergeable"),
            "mergeable_state": pull_request.get("mergeable_state"),
            "author_association": pull_request.get("author_association"),
            "additions": pull_request.get("additions"),
            "deletions": pull_request.get("deletions"),
            "changed_files": pull_request.get("changed_files"),
            "commits": pull_request.get("commits"),
            "comments": pull_request.get("comments"),
            "review_comments": pull_request.get("review_comments"),
            "labels": labels,
            "base": {
                "label": _get_nested(pull_request, "base.label"),
                "ref": _get_nested(pull_request, "base.ref"),
                "sha": _get_nested(pull_request, "base.sha"),
                "repo": {
                    "id": base_repo.get("id"),
                    "name": base_repo.get("name"),
                    "full_name": base_repo.get("full_name"),
                    "description": _truncate_text(base_repo.get("description"), 500),
                    "homepage": base_repo.get("homepage"),
                    "language": base_repo.get("language"),
                    "default_branch": base_repo.get("default_branch"),
                },
            },
            "head": {
                "label": _get_nested(pull_request, "head.label"),
                "ref": _get_nested(pull_request, "head.ref"),
                "sha": _get_nested(pull_request, "head.sha"),
                "repo": {
                    "id": head_repo.get("id"),
                    "name": head_repo.get("name"),
                    "full_name": head_repo.get("full_name"),
                    "description": _truncate_text(head_repo.get("description"), 500),
                    "homepage": head_repo.get("homepage"),
                    "language": head_repo.get("language"),
                    "default_branch": head_repo.get("default_branch"),
                },
            },
        },
        "participants": _truncate_list(
            _normalize_list_field(row.get("participants")), MAX_PARTICIPANTS
        ),
        "head_refs": _truncate_list(
            _normalize_list_field(row.get("head_refs")), MAX_HEAD_REFS
        ),
        "head_shas": _truncate_list(
            _normalize_list_field(row.get("head_shas")), MAX_HEAD_SHAS
        ),
        "landed_commit_sha": row.get("landed_commit_sha"),
        "landed_commit_shas": _truncate_list(
            _normalize_list_field(row.get("landed_commit_shas")), MAX_HEAD_SHAS
        ),
        "landed_inferred": row.get("landed_inferred"),
        "pr_body_versions": pr_body_versions,
        "events": events,
        "code": {
            "status": code.get("status"),
            "reason": _truncate_text(code.get("reason"), 500),
            "repo_full_name": code.get("repo_full_name"),
            "pr_number": code.get("pr_number"),
            "base_sha": code.get("base_sha"),
            "head_sha": code.get("head_sha"),
            "merge_base": code.get("merge_base"),
            "used_base_sha": code.get("used_base_sha"),
            "used_head_sha": code.get("used_head_sha"),
            "discussion_event_count": code.get("discussion_event_count"),
            "touched_files": _truncate_list(
                _normalize_list_field(code.get("touched_files")), MAX_TOUCHED_FILES
            ),
            "touched_files_truncated": code.get("touched_files_truncated"),
            "repo_files_at_base": _truncate_list(
                _normalize_list_field(code.get("repo_files_at_base")),
                MAX_REPO_FILES_AT_BASE,
            ),
            "repo_files_at_base_truncated": code.get("repo_files_at_base_truncated"),
            "compare_diff": _truncate_text(
                code.get("compare_diff"), MAX_COMPARE_DIFF_CHARS
            ),
            "compare_diff_truncated": code.get("compare_diff_truncated"),
            "commit_count": code.get("commit_count"),
            "commits_truncated": code.get("commits_truncated"),
            "commits": code_commits,
            "base_files_at_base": base_files_at_base,
            "base_files_at_base_truncated": code.get("base_files_at_base_truncated"),
        },
    }

    return _clean_none(pr_json)


def render_annotation_prompt(row: Dict[str, Any], prompt_template: str) -> str:
    pr_json_obj = _json_safe(build_pr_example_json(row))
    pr_json_text = json.dumps(pr_json_obj, indent=2, ensure_ascii=False)

    placeholder = "<PASTE_PR_JSON_HERE>"
    if placeholder in prompt_template:
        full_prompt = prompt_template.replace(placeholder, pr_json_text)
    else:
        full_prompt = f"{prompt_template.rstrip()}\n{pr_json_text}\n-------------"

    if len(full_prompt) > MAX_PROMPT_CHARS_FALLBACK:
        excess = len(full_prompt) - MAX_PROMPT_CHARS_FALLBACK
        if isinstance(pr_json_obj.get("code"), dict):
            current_diff = pr_json_obj["code"].get("compare_diff")
            pr_json_obj["code"]["compare_diff"] = _truncate_text(
                current_diff,
                max(4000, MAX_COMPARE_DIFF_CHARS - excess),
            )
        pr_json_text = json.dumps(pr_json_obj, indent=2, ensure_ascii=False)
        if placeholder in prompt_template:
            full_prompt = prompt_template.replace(placeholder, pr_json_text)
        else:
            full_prompt = f"{prompt_template.rstrip()}\n{pr_json_text}\n-------------"

    if len(full_prompt) > MAX_PROMPT_CHARS_FALLBACK:
        full_prompt = (
            full_prompt[:MAX_PROMPT_CHARS_FALLBACK] + "\n...[prompt truncated]"
        )

    return full_prompt


def try_parse_annotation_json(
    text: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "empty_response"

    text = text.strip()
    try:
        return json.loads(text), None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), None
        except Exception as e:
            return None, f"json_parse_error: {e}"

    return None, "no_json_object_found"


def normalize_annotation(
    annotation: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(annotation, dict):
        return None

    domain = annotation.get("domain")
    if not isinstance(domain, dict):
        domain = {}

    attributes = annotation.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    complexity = annotation.get("complexity")
    if not isinstance(complexity, dict):
        complexity = {}

    domain_scores_raw = domain.get("scores") or {}
    domain_scores = {}
    for label in DOMAIN_LABELS:
        try:
            domain_scores[label] = int(domain_scores_raw.get(label, 0))
        except Exception:
            domain_scores[label] = 0

    attribute_scores_raw = attributes.get("scores") or {}
    attribute_scores = {}
    for label in ATTRIBUTE_LABELS:
        try:
            attribute_scores[label] = int(attribute_scores_raw.get(label, 0))
        except Exception:
            attribute_scores[label] = 0

    complexity_scores_raw = complexity.get("scores") or {}
    complexity_scores = {}
    for label in COMPLEXITY_LABELS:
        try:
            complexity_scores[label] = int(complexity_scores_raw.get(label, 0))
        except Exception:
            complexity_scores[label] = 0

    domain_label = domain.get("label")
    if domain_label not in DOMAIN_LABELS:
        domain_label = "NONE"

    attribute_labels = attributes.get("labels")
    if not isinstance(attribute_labels, list):
        attribute_labels = []
    attribute_labels = [str(x) for x in attribute_labels if str(x) in ATTRIBUTE_LABELS]

    complexity_label = complexity.get("label")
    if complexity_label not in COMPLEXITY_LABELS:
        complexity_label = "NONE"

    domain_rationale = domain.get("rationale")
    if domain_rationale is None:
        domain_rationale = ""

    attr_rationale = attributes.get("rationale")
    if attr_rationale is None:
        attr_rationale = ""

    complexity_rationale = complexity.get("rationale")
    if complexity_rationale is None:
        complexity_rationale = ""

    return {
        "domain": {
            "label": domain_label,
            "scores": domain_scores,
            "rationale": str(domain_rationale),
        },
        "attributes": {
            "labels": attribute_labels,
            "scores": attribute_scores,
            "rationale": str(attr_rationale),
        },
        "complexity": {
            "label": complexity_label,
            "scores": complexity_scores,
            "rationale": str(complexity_rationale),
        },
    }


def find_parquet_files(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix != ".parquet":
            raise ValueError(f"Input file is not a parquet file: {input_path}")
        return [p]
    if p.is_dir():
        return sorted(x for x in p.rglob("*.parquet") if x.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_rows(parquet_path: Path) -> List[Dict[str, Any]]:
    return pq.read_table(parquet_path).to_pylist()


def build_llm(
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_num_batched_tokens: Optional[int],
    max_num_seqs: Optional[int],
    enable_prefix_caching: bool,
) -> LLM:
    """
    Throughput/stability choices:
    - Exposed tuning levers are only TP / MNS / MNBT.
    - Keep custom all-reduce disabled as a fixed runtime workaround for the TP startup crash.
    - Do NOT force eager mode; eager disables CUDA graphs and usually hurts performance.
    - Keep vLLM periodic stats logging enabled so we can compare runs.

    Metrics to watch in vLLM logs:
    - running / waiting requests:
        scheduler pressure and queueing
    - GPU KV cache usage:
        memory pressure; if pegged, reduce MNS / MNBT or increase TP
    - prompt throughput:
        prefill-side throughput
    - generation throughput:
        decode-side throughput; main KPI for long generations
    - prefix cache hit rate:
        indicates whether repeated prompt prefixes are helping
    """
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        "disable_custom_all_reduce": True,
        "disable_log_stats": False,
    }

    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs

    return LLM(**kwargs)


def annotate_prompts_vllm(
    llm: LLM,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
) -> List[str]:
    """
    Batch-level metrics printed here complement vLLM's own periodic logs.

    Meanings:
    - rows/s:
        completed examples per second
    - prompt_tok/s:
        input-side throughput (prefill)
    - gen_tok/s:
        output-side throughput (decode); main throughput KPI here
    - total_tok/s:
        combined throughput
    - avg_prompt_toks/row, avg_gen_toks/row:
        useful for comparing runs fairly when prompt/output lengths vary
    """
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]

    structured = StructuredOutputsParams(json=ANNOTATION_JSON_SCHEMA)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_completion_tokens,
        structured_outputs=structured,
    )

    started_at = time.perf_counter()
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    elapsed = max(time.perf_counter() - started_at, 1e-6)

    prompt_tokens = 0
    gen_tokens = 0
    for out in outputs:
        prompt_token_ids = getattr(out, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            prompt_tokens += len(prompt_token_ids)

        if out.outputs:
            token_ids = getattr(out.outputs[0], "token_ids", None)
            if token_ids is not None:
                gen_tokens += len(token_ids)

    total_tokens = prompt_tokens + gen_tokens
    logging.info(
        "batch rows=%d elapsed=%.2fs rows/s=%.2f prompt_toks=%d gen_toks=%d "
        "prompt_tok/s=%.1f gen_tok/s=%.1f total_tok/s=%.1f "
        "avg_prompt_toks/row=%.1f avg_gen_toks/row=%.1f",
        len(prompts),
        elapsed,
        len(prompts) / elapsed,
        prompt_tokens,
        gen_tokens,
        prompt_tokens / elapsed,
        gen_tokens / elapsed,
        total_tokens / elapsed,
        prompt_tokens / max(len(prompts), 1),
        gen_tokens / max(len(prompts), 1),
    )

    results = []
    for out in outputs:
        text = ""
        if out.outputs:
            text = out.outputs[0].text
        results.append(text)
    return results


def enrich_rows(
    row_batch: List[Dict[str, Any]],
    prompt_batch: List[str],
    raw_outputs: List[str],
) -> List[Dict[str, Any]]:
    enriched_rows = []
    for row, raw in zip(row_batch, raw_outputs):
        parsed, parse_error = try_parse_annotation_json(raw)
        annotation = normalize_annotation(parsed)

        entry = _json_safe(dict(row))
        entry["annotation"] = annotation
        entry["annotation_parse_error"] = parse_error
        enriched_rows.append(entry)
    return enriched_rows


def _annotation_arrow_type() -> pa.DataType:
    domain_scores_type = pa.struct([(label, pa.int64()) for label in DOMAIN_LABELS])
    attribute_scores_type = pa.struct(
        [(label, pa.int64()) for label in ATTRIBUTE_LABELS]
    )
    complexity_scores_type = pa.struct(
        [(label, pa.int64()) for label in COMPLEXITY_LABELS]
    )

    return pa.struct(
        [
            (
                "domain",
                pa.struct(
                    [
                        ("label", pa.string()),
                        ("scores", domain_scores_type),
                        ("rationale", pa.string()),
                    ]
                ),
            ),
            (
                "attributes",
                pa.struct(
                    [
                        ("labels", pa.list_(pa.string())),
                        ("scores", attribute_scores_type),
                        ("rationale", pa.string()),
                    ]
                ),
            ),
            (
                "complexity",
                pa.struct(
                    [
                        ("label", pa.string()),
                        ("scores", complexity_scores_type),
                        ("rationale", pa.string()),
                    ]
                ),
            ),
        ]
    )


def _append_or_replace_column(
    table: pa.Table, name: str, array: pa.Array | pa.ChunkedArray
) -> pa.Table:
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, array)
    return table.append_column(name, array)


def _tmp_output_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}_tmp{final_path.suffix}")


def _relative_output_path(
    input_root: Path, output_root: Path, input_file: Path
) -> Path:
    if input_root.is_file():
        return output_root / input_file.name
    return output_root / input_file.relative_to(input_root)


def _get_array_rank_info() -> Tuple[int, int]:
    rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    if world_size <= 0:
        world_size = 1
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid array rank info: SLURM_ARRAY_TASK_ID={rank}, "
            f"SLURM_ARRAY_TASK_COUNT={world_size}"
        )
    return rank, world_size


def _owned_files_for_rank(files: List[Path], rank: int, world_size: int) -> List[Path]:
    return [path for idx, path in enumerate(files) if idx % world_size == rank]


def _destroy_llm(llm: Optional[LLM]) -> None:
    if llm is None:
        return
    try:
        del llm
    except Exception:
        pass
    gc.collect()


def _is_retryable_engine_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    if "EngineDeadError" in text:
        return True
    if "RPC call to sample_tokens timed out" in text:
        return True
    if "EngineCore encountered an issue" in text:
        return True
    if "TimeoutError" in text:
        return True
    return False


def _process_one_parquet_file_to_path(
    parquet_path: Path,
    output_path: Path,
    prompt_template: str,
    llm: LLM,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    batch_size: int,
) -> Path:
    """
    batch_size is intentionally ignored.

    We use llm.llm_engine's configured max_num_seqs as the single Python-side
    batching unit so that MNS remains the only concurrency lever outside vLLM.
    """
    input_table = pq.read_table(parquet_path)
    rows = input_table.to_pylist()

    tokenizer = llm.get_tokenizer()
    configured_max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len
    allowed_input_tokens = configured_max_model_len - max_completion_tokens
    if allowed_input_tokens <= 0:
        raise ValueError(
            f"max_model_len ({configured_max_model_len}) must be greater than "
            f"max_completion_tokens ({max_completion_tokens})"
        )

    # Use the configured MNS from the engine; ignore external batch_size.
    configured_mns = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs

    num_rows = len(rows)
    annotations: List[Optional[Dict[str, Any]]] = [None] * num_rows
    annotation_parse_errors: List[Optional[str]] = [None] * num_rows
    prompt_was_truncated: List[bool] = [False] * num_rows
    prompt_tokens_original: List[Optional[int]] = [None] * num_rows
    prompt_tokens_kept: List[Optional[int]] = [None] * num_rows

    valid_indices: List[int] = []
    prompts: List[str] = []
    truncated_count = 0
    skipped_git_error_count = 0

    for row_idx, row in enumerate(rows):
        if row_has_git_error(row):
            skipped_git_error_count += 1
            annotation_parse_errors[row_idx] = "skipped_git_error"
            continue

        prompt = render_annotation_prompt(row, prompt_template)

        # Truncate prompt to fit the model context after reserving completion space.
        # This avoids the validation error you hit:
        # input_tokens > max_model_len - max_completion_tokens
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        original_tokens = len(input_ids)

        if original_tokens > allowed_input_tokens:
            input_ids = input_ids[:allowed_input_tokens]
            prompt = tokenizer.decode(input_ids)
            truncated_count += 1
            prompt_was_truncated[row_idx] = True
            prompt_tokens_original[row_idx] = original_tokens
            prompt_tokens_kept[row_idx] = len(input_ids)
        else:
            prompt_was_truncated[row_idx] = False
            prompt_tokens_original[row_idx] = original_tokens
            prompt_tokens_kept[row_idx] = original_tokens

        valid_indices.append(row_idx)
        prompts.append(prompt)

    logging.info(
        "Prepared file=%s rows=%d valid_rows=%d skipped_git_error_rows=%d truncated_prompts=%d max_model_len=%d max_completion_tokens=%d allowed_input_tokens=%d",
        parquet_path,
        len(rows),
        len(valid_indices),
        skipped_git_error_count,
        truncated_count,
        configured_max_model_len,
        max_completion_tokens,
        allowed_input_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = _tmp_output_path(output_path)
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    for start in range(0, len(valid_indices), configured_mns):
        batch_indices = valid_indices[start : start + configured_mns]
        prompt_batch = prompts[start : start + configured_mns]

        raw_outputs = annotate_prompts_vllm(
            llm=llm,
            prompts=prompt_batch,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
        )

        for row_idx, raw in zip(batch_indices, raw_outputs):
            parsed, parse_error = try_parse_annotation_json(raw)
            annotations[row_idx] = normalize_annotation(parsed)
            annotation_parse_errors[row_idx] = parse_error

    output_table = input_table
    output_table = _append_or_replace_column(
        output_table,
        "annotation",
        pa.array(annotations, type=_annotation_arrow_type()),
    )
    output_table = _append_or_replace_column(
        output_table,
        "annotation_parse_error",
        pa.array(annotation_parse_errors, type=pa.large_string()),
    )
    output_table = _append_or_replace_column(
        output_table,
        "prompt_was_truncated",
        pa.array(prompt_was_truncated, type=pa.bool_()),
    )
    output_table = _append_or_replace_column(
        output_table,
        "prompt_tokens_original",
        pa.array(prompt_tokens_original, type=pa.int64()),
    )
    output_table = _append_or_replace_column(
        output_table,
        "prompt_tokens_kept",
        pa.array(prompt_tokens_kept, type=pa.int64()),
    )

    pq.write_table(output_table, tmp_output_path)
    os.replace(tmp_output_path, output_path)
    logging.info("Wrote: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Top-level input parquet file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template-file", required=True)
    parser.add_argument("--model-name", required=True)

    # Only exposed throughput levers.
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--mns", type=int, default=128)
    parser.add_argument("--mnbt", type=int, default=65536)

    # Functional, not a throughput tuning lever.
    parser.add_argument("--max-completion-tokens", type=int, default=12000)

    # Retry knob.
    parser.add_argument(
        "--max-file-retries", type=int, default=DEFAULT_MAX_FILE_RETRIES
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_root = Path(args.input_path).resolve()
    output_root = Path(args.output_dir).resolve()

    with open(args.template_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    rank, world_size = _get_array_rank_info()
    all_input_files = find_parquet_files(args.input_path)
    owned_input_files = _owned_files_for_rank(all_input_files, rank, world_size)

    parquet_files_to_process: List[Tuple[Path, Path]] = []
    skipped_completed = 0

    for input_file in owned_input_files:
        output_file = _relative_output_path(input_root, output_root, input_file)
        if output_file.exists():
            skipped_completed += 1
            continue
        parquet_files_to_process.append((input_file, output_file))

    logging.info(
        "Rank ownership array_rank=%d array_world_size=%d total_parquet_files=%d owned_parquet_files=%d skipped_completed=%d parquet_to_process=%d",
        rank,
        world_size,
        len(all_input_files),
        len(owned_input_files),
        skipped_completed,
        len(parquet_files_to_process),
    )

    if not parquet_files_to_process:
        logging.info(
            "No parquet files assigned to this rank that still need processing."
        )
        return

    logging.info(
        "Run config model=%s tp=%d mns=%d mnbt=%d max_model_len=%d max_completion_tokens=%d max_file_retries=%d gmu=%.2f prefix_caching=%s",
        args.model_name,
        args.tp,
        args.mns,
        args.mnbt,
        MAX_MODEL_LEN,
        args.max_completion_tokens,
        args.max_file_retries,
        GPU_MEMORY_UTILIZATION,
        ENABLE_PREFIX_CACHING,
    )
    logging.info(
        "Watch vLLM logs for running/waiting requests, GPU KV cache usage, prompt throughput, generation throughput, and prefix cache hit rate."
    )

    llm: Optional[LLM] = None

    try:
        llm = build_llm(
            model_name=args.model_name,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=args.mnbt,
            max_num_seqs=args.mns,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
        )

        for parquet_path, output_path in parquet_files_to_process:
            attempt = 0
            while True:
                try:
                    _process_one_parquet_file_to_path(
                        parquet_path=parquet_path,
                        output_path=output_path,
                        prompt_template=prompt_template,
                        llm=llm,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_completion_tokens=args.max_completion_tokens,
                        batch_size=args.mns,  # ignored intentionally; MNS is the real batching control
                    )
                    break
                except Exception as e:
                    tmp_output_path = _tmp_output_path(output_path)
                    if tmp_output_path.exists():
                        tmp_output_path.unlink()

                    retryable = _is_retryable_engine_error(e)
                    if (not retryable) or attempt >= args.max_file_retries:
                        logging.exception(
                            "Failed file=%s attempt=%d retryable=%s",
                            parquet_path,
                            attempt + 1,
                            retryable,
                        )
                        raise

                    attempt += 1
                    logging.warning(
                        "Retrying file=%s after engine failure attempt=%d/%d error=%s",
                        parquet_path,
                        attempt,
                        args.max_file_retries,
                        repr(e),
                    )

                    _destroy_llm(llm)
                    llm = build_llm(
                        model_name=args.model_name,
                        tensor_parallel_size=args.tp,
                        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                        max_model_len=MAX_MODEL_LEN,
                        max_num_batched_tokens=args.mnbt,
                        max_num_seqs=args.mns,
                        enable_prefix_caching=ENABLE_PREFIX_CACHING,
                    )
    finally:
        _destroy_llm(llm)


if __name__ == "__main__":
    main()
