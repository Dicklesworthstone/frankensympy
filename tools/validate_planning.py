#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import tomllib
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []
WARNINGS: list[str] = []

REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md",
    "docs/CONSTITUTION.md",
    "docs/SOURCE_PROJECT_AUDIT.md",
    "docs/COMPATIBILITY_CONTRACT.md",
    "docs/OBJECT_MODEL_AND_IR.md",
    "docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md",
    "docs/EVIDENCE_PROOFS_AND_REWRITES.md",
    "docs/ALGORITHM_PORTFOLIOS.md",
    "docs/RUNTIME_BUDGETS_AND_DETERMINISM.md",
    "docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md",
    "docs/AGENT_NATIVE_PROTOCOL.md",
    "docs/CONFORMANCE_AND_BENCHMARKING.md",
    "docs/SECURITY_AND_RESOURCE_GOVERNANCE.md",
    "docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md",
    "docs/WORKSTREAM_GRAPH.md",
    "docs/FIRST_IMPLEMENTATION_CAMPAIGN.md",
    "docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md",
    "registries/compatibility_profiles.toml",
    "registries/evidence_classes.toml",
    "registries/workstreams.toml",
    "registries/claims.toml",
    "quality_gates.toml",
]

WORKSTREAM_ID_RE = re.compile(r"^WS\d{2}$")
MILESTONE_ID_RE = re.compile(r"^M\d+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
WORKSTREAM_REFERENCE_RE = re.compile(r"\bWS\d{2}\b")
MILESTONE_REFERENCE_RE = re.compile(r"\bM\d+\b")
PLACEHOLDERS = {"UNSET", "NOT_IMPLEMENTED", "TBD", "TODO"}


def error(message: str) -> None:
    ERRORS.append(message)


def warning(message: str) -> None:
    WARNINGS.append(message)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_toml(relative_path: str) -> dict:
    path = ROOT / relative_path
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        error(f"missing TOML file: {relative_path}")
    except tomllib.TOMLDecodeError as exc:
        error(f"invalid TOML in {relative_path}: {exc}")
    return {}


def require_string(record: dict, key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        error(f"{context}: `{key}` must be a non-empty string")
        return ""
    return value


def require_bool(record: dict, key: str, context: str) -> bool | None:
    value = record.get(key)
    if not isinstance(value, bool):
        error(f"{context}: `{key}` must be a boolean")
        return None
    return value


def is_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in PLACEHOLDERS
    if isinstance(value, list):
        return any(is_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(is_placeholder(item) for item in value.values())
    return False


def validate_required_files() -> None:
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            error(f"required planning artifact is missing: {relative_path}")


def validate_markdown_links() -> tuple[int, int]:
    markdown_files = sorted(ROOT.rglob("*.md"))
    checked_links = 0
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            raw_target = match.group(1).strip()
            if not raw_target:
                error(f"{rel(markdown_file)}: empty Markdown link target")
                continue
            if raw_target.startswith("<") and ">" in raw_target:
                target = raw_target[1 : raw_target.index(">")]
            else:
                target = raw_target.split(maxsplit=1)[0]
            target = unquote(target.strip())
            if not target or target.startswith("#"):
                continue
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("//"):
                continue
            target_path = parsed.path
            if not target_path:
                continue
            checked_links += 1
            if target_path.startswith("/"):
                resolved = ROOT / target_path.lstrip("/")
            else:
                resolved = markdown_file.parent / target_path
            try:
                resolved = resolved.resolve()
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                error(
                    f"{rel(markdown_file)}: relative link escapes repository root: {target}"
                )
                continue
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                error(
                    f"{rel(markdown_file)}:{line}: broken relative link `{target}` "
                    f"(resolved to {rel(resolved)})"
                )
    return len(markdown_files), checked_links


def validate_workstreams(data: dict) -> tuple[set[str], set[str], list[str]]:
    workstream_records = data.get("workstreams")
    milestone_records = data.get("milestones")
    if not isinstance(workstream_records, list):
        error("registries/workstreams.toml: `workstreams` must be an array of tables")
        workstream_records = []
    if not isinstance(milestone_records, list):
        error("registries/workstreams.toml: `milestones` must be an array of tables")
        milestone_records = []

    workstreams: dict[str, dict] = {}
    closure_gates: dict[str, str] = {}
    allowed_statuses = {
        "planned",
        "in_progress",
        "blocked",
        "validated",
        "closed",
        "retired",
    }
    for index, record in enumerate(workstream_records):
        context = f"workstreams[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        workstream_id = require_string(record, "id", context)
        require_string(record, "title", context)
        status = require_string(record, "status", context)
        milestone = require_string(record, "milestone", context)
        closure_gate = require_string(record, "closure_gate", context)
        require_bool(record, "independent_gate_owner_required", context)
        dependencies = record.get("dependencies")
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            error(f"{context}: `dependencies` must be an array of workstream IDs")
            dependencies = []
        if workstream_id and not WORKSTREAM_ID_RE.fullmatch(workstream_id):
            error(f"{context}: invalid workstream ID `{workstream_id}`")
        if workstream_id in workstreams:
            error(f"duplicate workstream ID: {workstream_id}")
        elif workstream_id:
            workstreams[workstream_id] = record
        if status and status not in allowed_statuses:
            error(f"{context}: unsupported status `{status}`")
        if milestone and not MILESTONE_ID_RE.fullmatch(milestone):
            error(f"{context}: invalid milestone ID `{milestone}`")
        if closure_gate:
            if not closure_gate.startswith("gate://"):
                error(f"{context}: closure gate must use `gate://`: {closure_gate}")
            prior = closure_gates.get(closure_gate)
            if prior is not None:
                error(
                    f"duplicate workstream closure gate `{closure_gate}` in "
                    f"{prior} and {workstream_id}"
                )
            closure_gates[closure_gate] = workstream_id

    milestone_records_by_id: dict[str, dict] = {}
    for index, record in enumerate(milestone_records):
        context = f"milestones[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        milestone_id = require_string(record, "id", context)
        require_string(record, "title", context)
        status = require_string(record, "status", context)
        required_workstreams = record.get("required_workstreams", [])
        required_checkpoints = record.get("required_checkpoints", [])
        checkpoint_workstreams = record.get("checkpoint_workstreams", [])
        if milestone_id and not MILESTONE_ID_RE.fullmatch(milestone_id):
            error(f"{context}: invalid milestone ID `{milestone_id}`")
        if milestone_id in milestone_records_by_id:
            error(f"duplicate milestone ID: {milestone_id}")
        elif milestone_id:
            milestone_records_by_id[milestone_id] = record
        if status and status not in allowed_statuses:
            error(f"{context}: unsupported status `{status}`")
        for key, values in (
            ("required_workstreams", required_workstreams),
            ("checkpoint_workstreams", checkpoint_workstreams),
        ):
            if not isinstance(values, list) or not all(
                isinstance(item, str) for item in values
            ):
                error(f"{context}: `{key}` must be an array of workstream IDs")
                continue
            for workstream_id in values:
                if workstream_id not in workstreams:
                    error(f"{context}: unknown {key} entry `{workstream_id}`")
        if not isinstance(required_checkpoints, list) or not all(
            isinstance(item, str) and item.startswith("gate://")
            for item in required_checkpoints
        ):
            error(f"{context}: `required_checkpoints` must contain only `gate://` IDs")
            required_checkpoints = []
        if not required_workstreams and not required_checkpoints:
            error(
                f"{context}: milestone must require closed workstreams or an explicit checkpoint"
            )
        if checkpoint_workstreams and not required_checkpoints:
            error(
                f"{context}: checkpoint workstreams require at least one checkpoint gate"
            )

    milestone_ids = set(milestone_records_by_id)
    for workstream_id, record in workstreams.items():
        milestone = record.get("milestone")
        if milestone not in milestone_ids:
            error(f"{workstream_id}: unknown milestone `{milestone}`")
        dependencies = record.get("dependencies", [])
        for dependency in dependencies:
            if dependency == workstream_id:
                error(f"{workstream_id}: workstream cannot depend on itself")
            elif dependency not in workstreams:
                error(f"{workstream_id}: unknown dependency `{dependency}`")

    milestone_ordinals = {
        milestone_id: int(milestone_id[1:])
        for milestone_id in milestone_ids
        if MILESTONE_ID_RE.fullmatch(milestone_id)
    }
    for workstream_id, record in workstreams.items():
        own_milestone = record.get("milestone")
        own_ordinal = milestone_ordinals.get(own_milestone)
        for dependency in record.get("dependencies", []):
            dependency_record = workstreams.get(dependency)
            if dependency_record is None:
                continue
            dependency_ordinal = milestone_ordinals.get(dependency_record.get("milestone"))
            if (
                own_ordinal is not None
                and dependency_ordinal is not None
                and dependency_ordinal > own_ordinal
            ):
                error(
                    f"{workstream_id} ({own_milestone}) depends on later workstream "
                    f"{dependency} ({dependency_record.get('milestone')})"
                )

    for milestone_id, record in milestone_records_by_id.items():
        milestone_ordinal = milestone_ordinals.get(milestone_id)
        for workstream_id in record.get("required_workstreams", []):
            workstream_record = workstreams.get(workstream_id)
            if workstream_record is None:
                continue
            workstream_ordinal = milestone_ordinals.get(workstream_record.get("milestone"))
            if (
                milestone_ordinal is not None
                and workstream_ordinal is not None
                and workstream_ordinal > milestone_ordinal
            ):
                error(
                    f"{milestone_id} requires full closure of later workstream "
                    f"{workstream_id} ({workstream_record.get('milestone')}); "
                    "use an explicit checkpoint instead"
                )

    indegree = {workstream_id: 0 for workstream_id in workstreams}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for workstream_id, record in workstreams.items():
        for dependency in record.get("dependencies", []):
            if dependency in workstreams:
                indegree[workstream_id] += 1
                outgoing[dependency].append(workstream_id)
    queue = deque(sorted(key for key, value in indegree.items() if value == 0))
    topological_order: list[str] = []
    while queue:
        current = queue.popleft()
        topological_order.append(current)
        for child in sorted(outgoing[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(topological_order) != len(workstreams):
        cyclic = sorted(key for key, value in indegree.items() if value > 0)
        error(f"workstream dependency graph contains a cycle involving: {', '.join(cyclic)}")

    policy = data.get("policy", {})
    if not isinstance(policy, dict):
        error("registries/workstreams.toml: `policy` must be a table")
    else:
        required_true = [
            "dependencies_must_be_acyclic",
            "unknown_workstream_fails_closed",
            "closure_requires_all_dependencies_closed",
            "closure_requires_gate_bundle",
            "closure_requires_claim_and_discrepancy_updates",
            "retired_ids_must_be_tombstoned",
            "structural_changes_single_writer",
        ]
        for key in required_true:
            if policy.get(key) is not True:
                error(f"workstream policy `{key}` must remain true")
        if policy.get("closure_by_prose_assertion_allowed") is not False:
            error("workstream policy must forbid closure by prose assertion")

    return set(workstreams), milestone_ids, topological_order


def validate_profiles(data: dict) -> set[str]:
    records = data.get("profiles")
    if not isinstance(records, list):
        error("registries/compatibility_profiles.toml: `profiles` must be an array")
        return set()
    profile_ids: set[str] = set()
    source_audit = (ROOT / "docs/SOURCE_PROJECT_AUDIT.md").read_text(encoding="utf-8")
    compatibility_contract = (ROOT / "docs/COMPATIBILITY_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    for index, record in enumerate(records):
        context = f"profiles[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        profile_id = require_string(record, "profile_id", context)
        kind = require_string(record, "kind", context)
        status = require_string(record, "status", context)
        upstream_commit = require_string(record, "upstream_commit", context)
        if profile_id in profile_ids:
            error(f"duplicate compatibility profile ID: {profile_id}")
        elif profile_id:
            profile_ids.add(profile_id)
        if upstream_commit and not COMMIT_RE.fullmatch(upstream_commit):
            error(f"{context}: upstream commit must be a 40-character lowercase SHA")
        if upstream_commit:
            if upstream_commit not in source_audit:
                error(f"{context}: upstream commit is absent from SOURCE_PROJECT_AUDIT.md")
            if upstream_commit not in compatibility_contract:
                error(f"{context}: upstream commit is absent from COMPATIBILITY_CONTRACT.md")
        if kind == "drift_observation":
            if status != "non_certifying":
                error(f"{context}: drift observations must be `non_certifying`")
            if record.get("certified_at_commit") != "NEVER":
                error(f"{context}: drift observations must set `certified_at_commit = NEVER`")
        if status == "certified" and is_placeholder(record):
            error(f"{context}: certified profile contains placeholder fields")
        if status == "certified" and record.get("gate_bundle") in {None, "UNSET"}:
            error(f"{context}: certified profile must name a gate bundle")

    policy = data.get("policy", {})
    if not isinstance(policy, dict):
        error("registries/compatibility_profiles.toml: `policy` must be a table")
    else:
        if policy.get("certification_requires_immutable_upstream") is not True:
            error("compatibility policy must require an immutable upstream")
        if policy.get("moving_head_can_certify") is not False:
            error("compatibility policy must forbid moving-head certification")
        if policy.get("upstream_runtime_fallback_allowed") is not False:
            error("compatibility policy must forbid upstream runtime fallback")
        if policy.get("profile_blending_allowed") is not False:
            error("compatibility policy must forbid profile blending")
        if policy.get("unknown_fields_fail_closed") is not True:
            error("compatibility policy must fail closed on unknown fields")
    return profile_ids


def validate_evidence(data: dict) -> set[str]:
    records = data.get("evidence_classes")
    outcomes = data.get("non_evidence_outcomes")
    if not isinstance(records, list):
        error("registries/evidence_classes.toml: `evidence_classes` must be an array")
        records = []
    if not isinstance(outcomes, list):
        error("registries/evidence_classes.toml: `non_evidence_outcomes` must be an array")
        outcomes = []
    evidence_ids: set[str] = set()
    for index, record in enumerate(records):
        context = f"evidence_classes[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        evidence_id = require_string(record, "id", context)
        require_bool(record, "terminal", context)
        require_bool(record, "mathematical", context)
        require_string(record, "requires_verifier", context)
        if evidence_id in evidence_ids:
            error(f"duplicate evidence class ID: {evidence_id}")
        elif evidence_id:
            evidence_ids.add(evidence_id)
    outcome_ids: set[str] = set()
    for index, record in enumerate(outcomes):
        context = f"non_evidence_outcomes[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        outcome_id = require_string(record, "id", context)
        if outcome_id in outcome_ids:
            error(f"duplicate non-evidence outcome ID: {outcome_id}")
        elif outcome_id:
            outcome_ids.add(outcome_id)
    overlap = evidence_ids & outcome_ids
    if overlap:
        error(f"IDs cannot be both evidence and non-evidence outcomes: {sorted(overlap)}")
    promotions = data.get("prohibited_promotions", {})
    if not isinstance(promotions, dict) or not promotions:
        error("evidence registry must define prohibited promotions")
    else:
        for key, value in promotions.items():
            if value is not True:
                error(f"prohibited evidence promotion `{key}` must remain true")
    policy = data.get("policy", {})
    if not isinstance(policy, dict):
        error("evidence registry policy must be a table")
    else:
        for key in (
            "unknown_class_fails_closed",
            "claim_schema_must_match",
            "context_and_domain_must_match",
            "verifier_version_must_match",
            "unverified_candidates_use_separate_cache_namespace",
            "stronger_evidence_query_rejects_weaker_entry",
        ):
            if policy.get(key) is not True:
                error(f"evidence policy `{key}` must remain true")
    return evidence_ids


def validate_claims(
    data: dict, workstream_ids: set[str], profile_ids: set[str]
) -> tuple[int, int]:
    records = data.get("claims")
    semantics = data.get("status_semantics")
    if not isinstance(records, list):
        error("registries/claims.toml: `claims` must be an array")
        records = []
    if not isinstance(semantics, dict) or not semantics:
        error("registries/claims.toml: `status_semantics` must be a non-empty table")
        semantics = {}
    allowed_statuses = set(semantics)
    claim_ids: set[str] = set()
    present_tense_count = 0
    for index, record in enumerate(records):
        context = f"claims[{index}]"
        if not isinstance(record, dict):
            error(f"{context}: record must be a table")
            continue
        claim_id = require_string(record, "id", context)
        require_string(record, "kind", context)
        require_string(record, "statement", context)
        status = require_string(record, "status", context)
        require_string(record, "minimum_evidence", context)
        gate = require_string(record, "gate", context)
        present_tense = require_bool(record, "present_tense_allowed", context)
        workstreams = record.get("workstreams")
        artifacts = record.get("evidence_artifacts")
        if claim_id in claim_ids:
            error(f"duplicate claim ID: {claim_id}")
        elif claim_id:
            claim_ids.add(claim_id)
        if status and status not in allowed_statuses:
            error(f"{context}: unknown claim status `{status}`")
        if gate and not gate.startswith("gate://"):
            error(f"{context}: gate must use `gate://`: {gate}")
        if not isinstance(workstreams, list) or not all(
            isinstance(item, str) for item in workstreams
        ):
            error(f"{context}: `workstreams` must be an array of IDs")
            workstreams = []
        for workstream_id in workstreams:
            if workstream_id not in workstream_ids:
                error(f"{context}: unknown workstream `{workstream_id}`")
        if not isinstance(artifacts, list) or not all(
            isinstance(item, str) for item in artifacts
        ):
            error(f"{context}: `evidence_artifacts` must be an array of strings")
            artifacts = []
        profile_id = record.get("profile_id")
        if profile_id is not None and profile_id not in profile_ids:
            error(f"{context}: unknown compatibility profile `{profile_id}`")
        if present_tense is True:
            present_tense_count += 1
            if status not in {"documented", "implemented_uncertified", "validated", "certified"}:
                error(
                    f"{context}: status `{status}` cannot permit a present-tense capability claim"
                )
        if status == "planned" and present_tense is not False:
            error(f"{context}: planned claims must set `present_tense_allowed = false`")
        if status in {"implemented_uncertified", "validated", "certified"}:
            if not artifacts or is_placeholder(artifacts):
                error(f"{context}: status `{status}` requires non-placeholder artifacts")
        if status == "documented":
            if not artifacts or is_placeholder(artifacts):
                error(f"{context}: documented claim requires existing documentation artifacts")
            for artifact in artifacts:
                artifact_path = ROOT / artifact
                if not artifact_path.is_file():
                    error(f"{context}: documentation artifact does not exist: {artifact}")
        if status == "certified" and not record.get("profile_id") and record.get("kind") == "compatibility":
            error(f"{context}: certified compatibility claim must name an immutable profile")

    policy = data.get("policy", {})
    if not isinstance(policy, dict):
        error("claims registry policy must be a table")
    else:
        required_true = [
            "unknown_claim_fails_closed",
            "present_tense_capability_requires_non_planned_status",
            "implemented_status_requires_artifact",
            "validated_status_requires_gate_bundle",
            "certified_status_requires_same_commit_gate_bundle",
            "performance_claim_requires_live_incumbent",
            "performance_claim_requires_semantic_admission",
            "mathematical_claim_requires_typed_claim_and_verifier",
            "compatibility_claim_requires_immutable_profile",
            "repair_claim_requires_decode_digest_schema_and_semantic_checks",
            "monitoring_claim_cannot_grant_mathematical_evidence",
            "retired_claim_ids_must_remain",
        ]
        for key in required_true:
            if policy.get(key) is not True:
                error(f"claim policy `{key}` must remain true")
    return len(claim_ids), present_tense_count


def validate_quality_gate_status(data: dict) -> None:
    status = data.get("registry_status")
    enforced = data.get("enforced")
    if status == "planning" and enforced is not False:
        error("quality_gates.toml: planning-stage target budgets must not be enforced")
    activation = data.get("activation_requirements", {})
    if not isinstance(activation, dict):
        error("quality_gates.toml: activation requirements must be a table")
        return
    if status == "planning":
        for key in (
            "monitor_implementation_exists",
            "measurement_schema_frozen",
            "representative_test_inventory_exists",
        ):
            if activation.get(key) is not False:
                error(f"quality_gates.toml: planning status requires `{key} = false`")
        if activation.get("same_commit_gate_bundle_required") is not True:
            error("quality_gates.toml: activation must require a same-commit gate bundle")
        if activation.get("claims_registry_update_required") is not True:
            error("quality_gates.toml: activation must require a claims-registry update")


def validate_package_status() -> None:
    cargo = load_toml("Cargo.toml")
    package = cargo.get("package", {})
    if not isinstance(package, dict):
        error("Cargo.toml: missing `[package]` table")
        return
    description = package.get("description", "")
    if not isinstance(description, str) or "planning" not in description.lower():
        error("Cargo.toml: package description must explicitly state planning-stage status")
    if package.get("publish") is not False:
        error("Cargo.toml: planning-stage package must set `publish = false`")
    metadata = package.get("metadata", {}).get("frankensympy", {})
    if not isinstance(metadata, dict):
        error("Cargo.toml: missing `[package.metadata.frankensympy]` table")
    else:
        if metadata.get("implementation_status") != "planning":
            error("Cargo.toml: implementation status must remain `planning`")
        for key in (
            "claims_registry",
            "compatibility_profile_registry",
            "workstream_registry",
        ):
            value = metadata.get(key)
            if not isinstance(value, str) or not (ROOT / value).is_file():
                error(f"Cargo.toml: metadata path `{key}` is missing or invalid")
    lib_source = (ROOT / "src/lib.rs").read_text(encoding="utf-8")
    if 'IMPLEMENTATION_STATUS: &str = "planning"' not in lib_source:
        error("src/lib.rs: machine-readable implementation status is not `planning`")


def validate_document_references(
    workstream_ids: set[str], milestone_ids: set[str]
) -> None:
    for markdown_file in sorted(ROOT.rglob("*.md")):
        text = markdown_file.read_text(encoding="utf-8")
        for workstream_id in sorted(set(WORKSTREAM_REFERENCE_RE.findall(text))):
            if workstream_id not in workstream_ids:
                error(f"{rel(markdown_file)}: references unknown workstream `{workstream_id}`")
        for milestone_id in sorted(set(MILESTONE_REFERENCE_RE.findall(text))):
            if milestone_id not in milestone_ids:
                error(f"{rel(markdown_file)}: references unknown milestone `{milestone_id}`")


def validate_status_language() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plan = (ROOT / "COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md").read_text(
        encoding="utf-8"
    )
    if "Current status: architecture and implementation plan." not in readme:
        error("README.md: missing exact current-status statement")
    if "runtime capabilities are not yet implemented or certified" not in plan:
        error("comprehensive plan: missing explicit non-implementation status")
    for forbidden in (
        "[![CI]",
        "[![Conformance]",
        "[![Safety]",
        "[![Determinism]",
    ):
        if forbidden in readme:
            error(f"README.md: uncertified badge marker remains: {forbidden}")


def _self_test_policy() -> dict:
    return {
        "unknown_claim_fails_closed": True,
        "present_tense_capability_requires_non_planned_status": True,
        "implemented_status_requires_artifact": True,
        "validated_status_requires_gate_bundle": True,
        "certified_status_requires_same_commit_gate_bundle": True,
        "performance_claim_requires_live_incumbent": True,
        "performance_claim_requires_semantic_admission": True,
        "mathematical_claim_requires_typed_claim_and_verifier": True,
        "compatibility_claim_requires_immutable_profile": True,
        "repair_claim_requires_decode_digest_schema_and_semantic_checks": True,
        "monitoring_claim_cannot_grant_mathematical_evidence": True,
        "retired_claim_ids_must_remain": True,
    }


def _self_test_status_semantics() -> dict:
    return {
        "planned": "not yet true",
        "documented": "specified but not built",
        "implemented_uncertified": "built without certification",
        "validated": "gate bundle passed",
        "certified": "certified on same commit",
    }


def run_self_test() -> int:
    """Negative-fixture harness: deliberately false claims MUST be rejected.

    Satisfies the WS00 acceptance item "a deliberate false claim fixture
    fails CI". Each fixture resets the error ledger, runs the real claims
    validator, and asserts both the rejection (negative cases) and the
    acceptance of an honest control claim.
    """
    failures: list[str] = []

    def run_fixture(name: str, claims: list[dict]) -> list[str]:
        global ERRORS, WARNINGS
        ERRORS = []
        WARNINGS = []
        data = {
            "status_semantics": _self_test_status_semantics(),
            "policy": _self_test_policy(),
            "claims": claims,
        }
        validate_claims(data, {"WS00"}, set())
        return list(ERRORS)

    # Control: an honest planned claim must produce zero errors. Guards
    # against a linter that merely rejects everything.
    control_errors = run_fixture(
        "control-honest-planned",
        [
            {
                "id": "CL-SELFTEST-CONTROL",
                "kind": "capability",
                "statement": "The workstream graph is defined in the registry.",
                "status": "planned",
                "minimum_evidence": "gate://ws00-governance",
                "gate": "gate://ws00-governance",
                "present_tense_allowed": False,
                "workstreams": ["WS00"],
                "evidence_artifacts": [],
            }
        ],
    )
    if control_errors:
        failures.append(f"honest control claim was rejected: {control_errors}")

    # Fixture 1: deliberate false claim - planned status asserting a
    # present-tense capability. This is exactly the lie the discipline
    # forbids; it must be rejected.
    false_claim_errors = run_fixture(
        "false-present-tense",
        [
            {
                "id": "CL-SELFTEST-FALSE",
                "kind": "capability",
                "statement": "FrankenSymPy certifiably outperforms SymPy today.",
                "status": "planned",
                "minimum_evidence": "gate://unmet",
                "gate": "gate://unmet",
                "present_tense_allowed": True,
                "workstreams": ["WS00"],
                "evidence_artifacts": [],
            }
        ],
    )
    if not any("present-tense" in message for message in false_claim_errors):
        failures.append(
            f"deliberate false present-tense claim was not rejected: {false_claim_errors}"
        )

    # Fixture 2: implemented-style status without evidence artifacts.
    uncertified_errors = run_fixture(
        "uncertified-without-artifacts",
        [
            {
                "id": "CL-SELFTEST-NOCERT",
                "kind": "capability",
                "statement": "The exact arithmetic substrate exists.",
                "status": "implemented_uncertified",
                "minimum_evidence": "gate://ws03-exact-arithmetic",
                "gate": "gate://ws03-exact-arithmetic",
                "present_tense_allowed": False,
                "workstreams": ["WS00"],
                "evidence_artifacts": [],
            }
        ],
    )
    if not any("artifact" in message for message in uncertified_errors):
        failures.append(
            f"claim without required artifacts was not rejected: {uncertified_errors}"
        )

    # Fixture 3: unknown workstream reference fails closed.
    unknown_ws_errors = run_fixture(
        "unknown-workstream",
        [
            {
                "id": "CL-SELFTEST-BADWS",
                "kind": "capability",
                "statement": "Planned work against a nonexistent workstream.",
                "status": "planned",
                "minimum_evidence": "gate://ws99-never",
                "gate": "gate://ws99-never",
                "present_tense_allowed": False,
                "workstreams": ["WS99"],
                "evidence_artifacts": [],
            }
        ],
    )
    if not any("unknown workstream" in message for message in unknown_ws_errors):
        failures.append(
            f"unknown workstream reference was not rejected: {unknown_ws_errors}"
        )

    if failures:
        print("claims-linter self-test FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("claims-linter self-test passed (false claims rejected, control accepted)")
    return 0


def main() -> int:
    validate_required_files()
    workstream_data = load_toml("registries/workstreams.toml")
    profile_data = load_toml("registries/compatibility_profiles.toml")
    evidence_data = load_toml("registries/evidence_classes.toml")
    claims_data = load_toml("registries/claims.toml")
    quality_data = load_toml("quality_gates.toml")

    workstream_ids, milestone_ids, topological_order = validate_workstreams(
        workstream_data
    )
    profile_ids = validate_profiles(profile_data)
    evidence_ids = validate_evidence(evidence_data)
    claim_count, present_tense_count = validate_claims(
        claims_data, workstream_ids, profile_ids
    )
    validate_quality_gate_status(quality_data)
    validate_package_status()
    validate_status_language()
    validate_document_references(workstream_ids, milestone_ids)
    markdown_count, link_count = validate_markdown_links()

    for message in WARNINGS:
        print(f"warning: {message}", file=sys.stderr)
    if ERRORS:
        print(f"planning integrity validation failed with {len(ERRORS)} error(s):", file=sys.stderr)
        for message in ERRORS:
            print(f"  - {message}", file=sys.stderr)
        return 1

    print("planning integrity validation passed")
    print(f"  required artifacts: {len(REQUIRED_FILES)}")
    print(f"  Markdown files: {markdown_count}")
    print(f"  checked relative links: {link_count}")
    print(f"  compatibility profiles: {len(profile_ids)}")
    print(f"  evidence classes: {len(evidence_ids)}")
    print(f"  workstreams: {len(workstream_ids)}")
    print(f"  milestones: {len(milestone_ids)}")
    print(f"  claims: {claim_count}")
    print(f"  present-tense claims currently permitted: {present_tense_count}")
    print(f"  topological order: {' -> '.join(topological_order)}")
    return 0

if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(run_self_test())
    raise SystemExit(main())
