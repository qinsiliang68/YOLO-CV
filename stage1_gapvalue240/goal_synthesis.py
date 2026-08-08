"""Final evidence synthesis for the frozen Stage1 GapValue 240-run study.

The functions in this module deliberately separate descriptive associations,
held-out prediction claims, causal claims, and capabilities that were never
collected.  They operate only on derived tables; canonical training outputs are
never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import beta


VALID_STATUSES = {
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INCONCLUSIVE",
    "NOT_TESTABLE",
}


def build_triad_execution_invariants(
    canonical_metrics: pd.DataFrame, exposure_audit: pd.DataFrame
) -> pd.DataFrame:
    """Join only the execution quantities that should be identical within a triad."""

    canonical_required = {"run_slot", "triad_id", "arm"}
    exposure_required = {"run_slot", "steps_per_epoch", "replay_total_rows"}
    missing_canonical = canonical_required - set(canonical_metrics.columns)
    missing_exposure = exposure_required - set(exposure_audit.columns)
    if missing_canonical:
        raise ValueError(f"canonical metrics missing columns: {sorted(missing_canonical)}")
    if missing_exposure:
        raise ValueError(f"exposure audit missing columns: {sorted(missing_exposure)}")
    if canonical_metrics["run_slot"].duplicated().any() or exposure_audit[
        "run_slot"
    ].duplicated().any():
        raise ValueError("run_slot must be unique in canonical and exposure tables")
    for triad_id, group in canonical_metrics.groupby("triad_id", sort=False):
        if set(group["arm"].astype(str)) != {"T", "R1", "R2"} or len(group) != 3:
            raise ValueError(f"Triad {triad_id} must contain exactly T/R1/R2")
    invariant_columns = [
        column
        for column in exposure_audit.columns
        if column
        in {
            "steps_per_epoch",
            "replay_total_rows",
            "effective_batch_size",
            "epoch_samples",
            "completed_epochs",
            "optimizer_steps_total",
            "total_replay_exposures",
        }
        or column.startswith(
            (
                "optimizer_steps_to_",
                "optimizer_steps_after_",
                "replay_exposures_to_",
                "replay_exposures_after_",
                "lr_step_integral_pg",
                "lr_replay_integral_pg",
            )
        )
    ]
    joined = canonical_metrics[["run_slot", "triad_id", "arm"]].merge(
        exposure_audit[["run_slot", *invariant_columns]],
        on="run_slot",
        how="left",
        validate="one_to_one",
    )
    if joined[invariant_columns].isna().any().any():
        raise ValueError("Execution invariant join contains missing values")
    return joined.sort_values(["triad_id", "arm"], ignore_index=True)


def _truth(facts: Mapping[str, object], name: str) -> bool:
    if name not in facts:
        raise ValueError(f"Missing final-evidence fact: {name}")
    return bool(facts[name])


def _integer(facts: Mapping[str, object], name: str) -> int:
    if name not in facts:
        raise ValueError(f"Missing final-evidence fact: {name}")
    return int(facts[name])


def build_final_hypothesis_registry(facts: Mapping[str, object]) -> pd.DataFrame:
    """Classify every registered claim using explicit, auditable facts.

    ``SUPPORTED`` never implies causality here.  The output records the exact
    evidence layer so that a descriptive association cannot silently become an
    unseen-seed prediction claim.
    """

    late_full = _truth(facts, "late_loss_full_q_below_005_both_controls")
    late_discovery = _truth(
        facts, "late_loss_discovery_q_below_005_both_controls"
    )
    late_same_selection = _truth(facts, "late_loss_same_selection_supported")
    late_predictive = _truth(facts, "late_loss_unseen_seed_rule_supported")
    lr_kink = _truth(facts, "lr_schedule_has_150_160_kink")
    equal_lr_exposure = _truth(facts, "within_triad_lr_and_exposure_equal")
    checkpoint_robust = _truth(facts, "checkpoint_seed_ci_excludes_zero")
    raw_safe = _integer(facts, "raw_dual_safe_triads")
    raw_full = _integer(facts, "raw_full_frontier_dual_triads")
    total_triads = _integer(facts, "total_triads")
    reversal_groups = _integer(facts, "same_selection_reversal_groups")
    reversal_triads = _integer(facts, "same_selection_reversal_triads")
    phase_c_successes = _integer(facts, "phase_c_successes")
    phase_c_total = _integer(facts, "phase_c_total")
    joint_successes = _integer(facts, "joint_rule_phase_c_successes")
    joint_total = _integer(facts, "joint_rule_phase_c_total")

    if total_triads <= 0:
        raise ValueError("total_triads must be positive")
    if not 0 <= raw_safe <= total_triads or not 0 <= raw_full <= total_triads:
        raise ValueError("Raw-frontier counts are outside the triad range")

    records: list[dict[str, Any]] = []

    def add(
        hypothesis_id: str,
        topic: str,
        statement: str,
        status: str,
        evidence_layer: str,
        evidence: str,
        limitation: str,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid hypothesis status: {status}")
        records.append(
            {
                "hypothesis_id": hypothesis_id,
                "topic": topic,
                "hypothesis": statement,
                "status": status,
                "evidence_layer": evidence_layer,
                "evidence": evidence,
                "limitation": limitation,
                "causal_claim_allowed": False,
            }
        )

    add(
        "H01_LATE_EXTRA_FIT_ASSOCIATION",
        "memorization;replay_overfit;early_stopping",
        "Dual-improvement runs show less treatment-specific late train-loss reduction than dual-harm runs.",
        "SUPPORTED" if late_full else "INCONCLUSIVE",
        "DESCRIPTIVE_ASSOCIATION",
        (
            "Both R1 and R2 pass the all-80 seed-stratified FDR gate."
            if late_full
            else "The all-80 paired/FDR evidence does not pass for both controls."
        ),
        (
            "Discovery-only FDR passes both controls."
            if late_discovery
            else "Discovery-only FDR does not pass 0.05; this is not an independently confirmed predictor."
        ),
    )
    records[-1]["limitation"] += (
        " The association persists within exact selection blocks."
        if late_same_selection
        else " It disappears in exact-selection blocked reversal comparisons."
    )
    add(
        "H02_LATE_EXTRA_FIT_EARLY_WARNING",
        "memorization;early_stopping",
        "A pre-outcome late-fit rule predicts dual improvement on unseen training seeds.",
        "SUPPORTED" if late_predictive else "NOT_SUPPORTED",
        "HELD_OUT_PREDICTION",
        "Leakage-safe leave-seed/Phase-C validation result.",
        "Training loss association alone is not a deployable selection rule.",
    )
    add(
        "H03_LR_SCHEDULE_KINK",
        "learning_rate_stability",
        "A learning-rate discontinuity around epochs 150-160 causes the observed cohort split.",
        "SUPPORTED" if lr_kink else "NOT_SUPPORTED",
        "EXACT_SCHEDULE_AUDIT",
        "Per-epoch pg0-pg7 schedules were checked for slope discontinuities.",
        "An absent scheduler kink does not identify the causal interaction.",
    )
    add(
        "H04_SELECTION_LR_EXPOSURE_INTERACTION",
        "learning_rate_stability;replay_overfit",
        "Selection composition interacts with cumulative replay exposure under a continuing non-zero LR.",
        "INCONCLUSIVE" if equal_lr_exposure else "NOT_SUPPORTED",
        "MECHANISM_INFERENCE",
        "Within triads, LR, steps and replay-slot counts are equal across T/R1/R2.",
        "Per-step gradients, minibatch order and augmentation realizations were not retained.",
    )
    add(
        "H05_LAYERWISE_CHECKPOINT_DRIFT",
        "parameter_drift",
        "Best-to-last layerwise parameter drift robustly separates dual improvement from dual harm.",
        "SUPPORTED" if checkpoint_robust else "INCONCLUSIVE",
        "CHECKPOINT_ASSOCIATION",
        "Initial/best/last EMA tensor and block drift with seed/condition bootstrap.",
        "Only three checkpoint states exist and best-epoch ties remain ambiguous.",
    )
    add(
        "H06_STATIC_SELECTION_STABILITY",
        "training_dynamics;algorithmic_stability",
        "A fixed treatment sample set has a stable sign across training seeds.",
        "NOT_SUPPORTED" if reversal_groups > 0 else "INCONCLUSIVE",
        "SAME_SELECTION_CROSS_SEED",
        f"{reversal_groups} exact selections spanning {reversal_triads} triads reverse between good and harmful outcomes.",
        "The experiment does not isolate every source of initialization stochasticity.",
    )
    add(
        "H07_RAW_SAFE_FRONTIER_WIDESPREAD",
        "neyman_pearson;partial_auc",
        "Current treatment rules broadly dominate both controls on the raw-score safe FN frontier.",
        "SUPPORTED" if raw_safe >= max(1, int(0.8 * total_triads)) else "NOT_SUPPORTED",
        "RAW_SCORE_FRONTIER",
        f"{raw_safe}/{total_triads} triads dominate both controls for every raw FN budget 0-95; {raw_full}/{total_triads} do so over the full FN range.",
        "val_op is an internal benchmark, not an untouched external set.",
    )
    add(
        "H08_LEARNABLE_NOT_PERSISTENT",
        "forgetting;dataset_cartography;aum;grand_el2n",
        "Low-forgetting, finally learned replay composition is seed-robustly superior to persistent-hard composition.",
        "SUPPORTED" if _truth(facts, "learnability_seed_robust") else "INCONCLUSIVE",
        "SELECTION_COMPOSITION",
        "Forgetting, late persistence, final correctness, bucket and margin proxies were compared.",
        "The available 0.5-based dynamics are misaligned with the operational threshold, and true GraNd is absent.",
    )
    add(
        "H09_GUARD_AND_BUDGET_GENERALIZATION",
        "replay_overfit;partial_auc",
        "Small-budget normal replay and learnable defect guard generalize robustly across seeds.",
        (
            "SUPPORTED"
            if _truth(facts, "guard_seed_robust")
            and _truth(facts, "budget_response_seed_robust")
            else "INCONCLUSIVE"
        ),
        "METHOD_AND_BUDGET_ABLATION",
        "All frozen Phase-A/B condition, budget and guard results were included.",
        "Three seeds per condition and Phase-B machine confounding prevent a stable causal conclusion.",
    )
    observed_joint_rate = joint_successes / joint_total if joint_total else 0.0
    add(
        "H10_UNSEEN_SEED_80PCT",
        "algorithmic_stability;model_selection",
        "A predeclared rule beats both R1 and R2 on at least 80% of unseen training seeds.",
        "SUPPORTED" if late_predictive and observed_joint_rate >= 0.8 else "NOT_SUPPORTED",
        "UNSEEN_SEED_EXTERNAL_VALIDATION",
        (
            f"Fixed A02: {phase_c_successes}/{phase_c_total} Phase-C successes; "
            f"joint rule: {joint_successes}/{joint_total}."
        ),
        "Only five completely new Phase-C seeds exist; even 5/5 cannot prove a one-sided 95% lower bound above 80%.",
    )
    add(
        "H11_TRUE_GRADIENT_ALIGNMENT",
        "grand_el2n;influence;tracin;gradient_matching",
        "True per-sample gradient magnitude/alignment predicts operational sample value.",
        "INCONCLUSIVE" if _truth(facts, "gradients_collected") else "NOT_TESTABLE",
        "MISSING_CAPABILITY",
        "Gradient-related source fields were explicitly audited.",
        "Per-sample gradients and gradient embeddings were not collected.",
    )
    add(
        "H12_EPOCH150_PARAMETER_STATE",
        "parameter_drift;mode_connectivity",
        "The exact epoch-150 weight state explains the 150-160 transition.",
        "INCONCLUSIVE" if _truth(facts, "epoch150_checkpoint_collected") else "NOT_TESTABLE",
        "MISSING_CAPABILITY",
        "Checkpoint inventory was audited.",
        "Only initial, stripped best and final resumable checkpoints were retained.",
    )
    add(
        "H13_REPLAY_VS_NO_REPLAY",
        "replay_overfit",
        "Replay itself is superior to training with no replay.",
        "INCONCLUSIVE" if _truth(facts, "no_replay_arm_collected") else "NOT_TESTABLE",
        "MISSING_CONTROL",
        "The frozen matrix was audited for all arms.",
        "The 240-run study contains T/R1/R2 replay arms and no no-replay arm.",
    )
    add(
        "H14_EXTERNAL_GENERALIZATION",
        "model_selection;neyman_pearson",
        "The observed val_op mechanism generalizes to an untouched deployment distribution.",
        "INCONCLUSIVE" if _truth(facts, "blind_external_collected") else "NOT_TESTABLE",
        "MISSING_EVALUATION_SET",
        "Evaluation split provenance was audited.",
        "No blind or external test was supplied.",
    )
    return pd.DataFrame(records)


_TOPIC_TO_HYPOTHESES: dict[str, tuple[str, ...]] = {
    "memorization": ("H01_LATE_EXTRA_FIT_ASSOCIATION", "H02_LATE_EXTRA_FIT_EARLY_WARNING"),
    "forgetting": ("H08_LEARNABLE_NOT_PERSISTENT",),
    "dataset_cartography": ("H08_LEARNABLE_NOT_PERSISTENT",),
    "aum": ("H08_LEARNABLE_NOT_PERSISTENT",),
    "rho_loss": ("H08_LEARNABLE_NOT_PERSISTENT",),
    "grand_el2n": ("H08_LEARNABLE_NOT_PERSISTENT", "H11_TRUE_GRADIENT_ALIGNMENT"),
    "influence": ("H11_TRUE_GRADIENT_ALIGNMENT",),
    "tracin": ("H11_TRUE_GRADIENT_ALIGNMENT",),
    "gradient_matching": ("H11_TRUE_GRADIENT_ALIGNMENT",),
    "replay_overfit": ("H01_LATE_EXTRA_FIT_ASSOCIATION", "H04_SELECTION_LR_EXPOSURE_INTERACTION", "H13_REPLAY_VS_NO_REPLAY"),
    "learning_rate_stability": ("H03_LR_SCHEDULE_KINK", "H04_SELECTION_LR_EXPOSURE_INTERACTION"),
    "edge_of_stability": ("H03_LR_SCHEDULE_KINK", "H04_SELECTION_LR_EXPOSURE_INTERACTION"),
    "parameter_drift": ("H05_LAYERWISE_CHECKPOINT_DRIFT", "H12_EPOCH150_PARAMETER_STATE"),
    "swa_mode_connectivity": ("H12_EPOCH150_PARAMETER_STATE",),
    "early_stopping": ("H01_LATE_EXTRA_FIT_ASSOCIATION", "H02_LATE_EXTRA_FIT_EARLY_WARNING"),
    "neyman_pearson": ("H07_RAW_SAFE_FRONTIER_WIDESPREAD", "H14_EXTERNAL_GENERALIZATION"),
    "partial_auc": ("H07_RAW_SAFE_FRONTIER_WIDESPREAD",),
}


def build_literature_result_matrix(
    literature: pd.DataFrame, hypotheses: pd.DataFrame
) -> pd.DataFrame:
    """Join every preregistered primary source to an empirical result boundary."""

    required = {"evidence_id", "topic", "testability_status"}
    missing = required - set(literature.columns)
    if missing:
        raise ValueError(f"literature missing columns: {sorted(missing)}")
    required_h = {"hypothesis_id", "status", "evidence", "limitation"}
    missing_h = required_h - set(hypotheses.columns)
    if missing_h:
        raise ValueError(f"hypotheses missing columns: {sorted(missing_h)}")
    lookup = hypotheses.set_index("hypothesis_id")
    records: list[dict[str, object]] = []
    for row in literature.to_dict(orient="records"):
        topic = str(row["topic"])
        hypothesis_ids = _TOPIC_TO_HYPOTHESES.get(topic)
        if not hypothesis_ids:
            raise ValueError(f"No empirical hypothesis mapping for literature topic: {topic}")
        absent = [item for item in hypothesis_ids if item not in lookup.index]
        if absent:
            raise ValueError(f"Missing mapped hypotheses: {absent}")
        source_not_testable = str(row["testability_status"]) == "NOT_TESTABLE"
        statuses = [str(lookup.loc[item, "status"]) for item in hypothesis_ids]
        if source_not_testable:
            result_status = "NOT_TESTABLE"
        elif "SUPPORTED" in statuses:
            result_status = "SUPPORTED"
        elif all(value == "NOT_SUPPORTED" for value in statuses):
            result_status = "NOT_SUPPORTED"
        else:
            result_status = "INCONCLUSIVE"
        missing_capability = str(row.get("missing_required_capabilities", ""))
        boundary = (
            f"Not testable in this study: {missing_capability or 'required capability absent'}."
            if source_not_testable
            else " | ".join(str(lookup.loc[item, "limitation"]) for item in hypothesis_ids)
        )
        records.append(
            {
                **row,
                "mapped_hypothesis_ids": ";".join(hypothesis_ids),
                "mapped_hypothesis_statuses": ";".join(statuses),
                "study_result_status": result_status,
                "study_result_evidence": " | ".join(
                    str(lookup.loc[item, "evidence"]) for item in hypothesis_ids
                ),
                "study_result_boundary": boundary,
            }
        )
    return pd.DataFrame(records)


def _one_sided_lower(successes: int, total: int, confidence: float) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, total - successes + 1))


def minimum_unseen_seed_evidence(
    *,
    target_rate: float = 0.8,
    confidence: float = 0.95,
    allowed_failures: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """Minimum trials whose one-sided Clopper-Pearson lower bound exceeds target."""

    if not 0 < target_rate < 1 or not 0 < confidence < 1:
        raise ValueError("target_rate and confidence must lie in (0, 1)")
    records = []
    for failures in allowed_failures:
        if failures < 0:
            raise ValueError("allowed_failures must be non-negative")
        total = max(1, failures + 1)
        while total < 1_000_000:
            successes = total - failures
            lower = _one_sided_lower(successes, total, confidence)
            if lower > target_rate:
                records.append(
                    {
                        "target_rate": target_rate,
                        "confidence": confidence,
                        "allowed_failures": int(failures),
                        "minimum_total": int(total),
                        "required_successes": int(successes),
                        "one_sided_lower_bound": lower,
                    }
                )
                break
            total += 1
        else:
            raise RuntimeError("Unable to find a finite binomial evidence requirement")
    return pd.DataFrame(records)


def _common_fdr_cutoff(frame: pd.DataFrame, *, alpha: float = 0.05) -> bool:
    required = {"control", "feature", "q_value_bh"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"late-dynamics FDR table missing columns: {sorted(missing)}")
    subset = frame.loc[
        frame["feature"].astype(str).str.startswith("extra_train_loss_decline__at_")
        & pd.to_numeric(frame["q_value_bh"], errors="coerce").lt(alpha)
    ].copy()
    if subset.empty:
        return False
    subset["cutoff"] = subset["feature"].astype(str).str.rsplit("_", n=1).str[-1]
    return any(
        set(group["control"].astype(str)) >= {"R1", "R2"}
        for _, group in subset.groupby("cutoff")
    )


def _ci_excludes_zero(low: pd.Series, high: pd.Series) -> pd.Series:
    low_numeric = pd.to_numeric(low, errors="coerce")
    high_numeric = pd.to_numeric(high, errors="coerce")
    return ((low_numeric > 0) & (high_numeric > 0)) | (
        (low_numeric < 0) & (high_numeric < 0)
    )


def extract_final_evidence_facts(report_root: str | Path) -> dict[str, object]:
    """Recover final hypothesis inputs directly from generated evidence tables."""

    root = Path(report_root)
    tables = root / "tables"
    audit = root / "audit"

    def csv(name: str) -> pd.DataFrame:
        path = tables / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def json_file(name: str) -> dict[str, Any]:
        path = tables / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    full_fdr = csv("targeted_late_dynamics_permutation_fdr.csv")
    discovery_fdr = csv("targeted_late_dynamics_discovery_permutation_fdr.csv")
    joint = csv("joint_prediction_summaries.csv")
    outcomes = csv("triad_outcomes_80.csv")
    raw = json_file("raw_frontier_analysis_summary.json")
    selection = json_file("selection_mechanism_summary.json")
    reversal = json_file("reversal_analysis_summary.json")
    lr = csv("learning_rate_active_group_audit.csv")
    checkpoint = csv("checkpoint_cohort_contrasts.csv")
    invariants = csv("triad_execution_invariants.csv")
    ledger_path = audit / "DATA_USAGE_LEDGER_REFINED.csv"
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    ledger = pd.read_csv(ledger_path)

    external = joint.loc[
        joint["validation_scheme"].astype(str).eq("PHASE_C_EXTERNAL_FALSIFICATION")
    ].copy()
    external_selected = pd.to_numeric(
        external.get("selected_n", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    external_confirmed = external.get(
        "confirmed_above_target", pd.Series(False, index=external.index)
    ).astype(bool)
    late_predictive = bool((external_selected.gt(0) & external_confirmed).any())

    phase_c = outcomes.loc[outcomes["phase"].astype(str).eq("C")]
    phase_c_successes = int(phase_c["dual_improvement"].astype(bool).sum())
    phase_c_total = int(len(phase_c))
    joint_external_total = int(
        pd.to_numeric(external.get("n", pd.Series(dtype=float)), errors="coerce").max()
    ) if len(external) else 0
    joint_external_successes = int(
        pd.to_numeric(
            external.get("selected_successes", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0).max()
    ) if len(external) else 0

    max_second = pd.to_numeric(
        lr["max_abs_second_difference_after_120"], errors="raise"
    ).abs().max()
    lr_kink = bool(max_second > 1e-10)

    seed_robust = _ci_excludes_zero(
        checkpoint["seed_bootstrap_ci_low"], checkpoint["seed_bootstrap_ci_high"]
    )
    condition_robust = _ci_excludes_zero(
        checkpoint["condition_bootstrap_ci_low"],
        checkpoint["condition_bootstrap_ci_high"],
    )
    checkpoint_robust = bool((seed_robust & condition_robust).any())

    invariant_columns = [
        column
        for column in invariants.columns
        if column not in {"triad_id", "arm", "run_slot"}
    ]
    equal_lr_exposure = bool(
        invariant_columns
        and all(
            invariants.groupby("triad_id", dropna=False)[column].nunique(
                dropna=False
            ).le(1).all()
            for column in invariant_columns
        )
    )

    condition_path = tables / "condition_performance_summary.csv"
    if condition_path.is_file():
        conditions = pd.read_csv(condition_path)
        all_seed_success = (
            pd.to_numeric(conditions["dual_improvement"], errors="coerce")
            .eq(pd.to_numeric(conditions["triads"], errors="coerce"))
            & pd.to_numeric(conditions["dual_harm"], errors="coerce").eq(0)
        )
        robust_rows = conditions.loc[all_seed_success]
        learnability_seed_robust = bool(len(robust_rows))
        guard_seed_robust = bool(
            robust_rows["condition_slot"].astype(str).str.startswith("B").any()
        )
        strict = conditions.loc[
            conditions["method"].astype(str).eq("GapCritical-Strict")
            & conditions["discovery_or_confirmation"].astype(str).eq("discovery")
        ]
        budget_response_seed_robust = bool(
            len(strict) >= 3
            and (
                pd.to_numeric(strict["dual_improvement"], errors="coerce")
                == pd.to_numeric(strict["triads"], errors="coerce")
            ).all()
        )
    else:
        learnability_seed_robust = False
        guard_seed_robust = False
        budget_response_seed_robust = False

    def not_collected(field: str) -> bool:
        rows = ledger.loc[ledger["field_path"].astype(str).eq(field)]
        if rows.empty:
            raise ValueError(f"Refined field ledger lacks capability row: {field}")
        return rows["usage_status"].astype(str).isin(
            {"NOT_TESTABLE", "NOT_COLLECTED", "DOCUMENTED_MISSING"}
        ).all()

    return {
        "late_loss_full_q_below_005_both_controls": _common_fdr_cutoff(full_fdr),
        "late_loss_discovery_q_below_005_both_controls": _common_fdr_cutoff(
            discovery_fdr
        ),
        "late_loss_same_selection_supported": bool(
            float(
                reversal["focus_epoch_evidence"][
                    "extra_train_loss_decline_at_200"
                ]["fdr_q_global"]
            )
            < 0.05
        ),
        "late_loss_unseen_seed_rule_supported": late_predictive,
        "lr_schedule_has_150_160_kink": lr_kink,
        "within_triad_lr_and_exposure_equal": equal_lr_exposure,
        "checkpoint_seed_ci_excludes_zero": checkpoint_robust,
        "raw_dual_safe_triads": int(
            raw["raw_dual_control_safe_frontier_dominant_triads"]
        ),
        "raw_full_frontier_dual_triads": int(
            raw["raw_dual_control_full_frontier_dominant_triads"]
        ),
        "total_triads": int(raw["canonical_triads"]),
        "same_selection_reversal_groups": int(
            selection["same_selection_reversal_digests"]
        ),
        "same_selection_reversal_triads": int(
            selection["same_selection_reversal_triads"]
        ),
        "phase_c_successes": phase_c_successes,
        "phase_c_total": phase_c_total,
        "joint_rule_phase_c_successes": joint_external_successes,
        "joint_rule_phase_c_total": joint_external_total,
        "gradients_collected": not bool(selection["gradient_fields_not_collected"]),
        "epoch150_checkpoint_collected": not not_collected("epoch_150_checkpoint"),
        "no_replay_arm_collected": not not_collected("no_replay_arm"),
        "blind_external_collected": not not_collected("blind_or_external_test"),
        "learnability_seed_robust": learnability_seed_robust,
        "guard_seed_robust": guard_seed_robust,
        "budget_response_seed_robust": budget_response_seed_robust,
    }


def completion_gate_audit(
    output_dir: str | Path,
    gates: Mapping[str, object],
    *,
    required_files: Sequence[str],
) -> pd.DataFrame:
    """Verify the non-negotiable numeric and artifact completion gates."""

    expected = {
        "canonical_runs": 240,
        "triads": 80,
        "paired_comparisons": 160,
        "epoch_rows": 48_000,
        "UNREVIEWED": 0,
        "UNCLASSIFIED": 0,
        "SILENTLY_DROPPED": 0,
    }
    records: list[dict[str, object]] = []
    errors = []
    for gate, expected_value in expected.items():
        actual = int(gates.get(gate, -1))
        passed = actual == expected_value
        records.append(
            {
                "gate": gate,
                "expected": expected_value,
                "actual": actual,
                "passed": passed,
                "evidence_type": "numeric_gate",
            }
        )
        if not passed:
            errors.append(f"{gate}={actual}, expected {expected_value}")
    root = Path(output_dir)
    for relative in required_files:
        exists = (root / relative).is_file()
        records.append(
            {
                "gate": f"required_file:{relative}",
                "expected": "present",
                "actual": "present" if exists else "missing",
                "passed": exists,
                "evidence_type": "artifact_gate",
            }
        )
        if not exists:
            errors.append(f"required_file:{relative}=missing")
    if errors:
        raise ValueError("Completion gate failed: " + "; ".join(errors))
    return pd.DataFrame(records)
