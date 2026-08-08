"""Primary-source evidence map for the 240-run mechanism analysis.

The literature is used to define falsifiable hypotheses and measurement
requirements.  It is not used to promote an unavailable quantity (for example,
an influence function without gradients and Hessian information) into an
observed Stage1 feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd


class LiteratureEvidenceError(RuntimeError):
    """Raised when the evidence registry is incomplete or misleading."""


DIRECTLY_TESTABLE = "DIRECTLY_TESTABLE"
PARTIALLY_TESTABLE = "PARTIALLY_TESTABLE"
NOT_TESTABLE = "NOT_TESTABLE"
CONTEXT_ONLY = "CONTEXT_ONLY"


DEFAULT_COLLECTED_CAPABILITIES = frozenset(
    {
        "sample_labels",
        "per_sample_oof_probability_trajectory",
        "binary_logit_margin_reconstructable",
        "selection_training_dynamic_summaries",
        "per_epoch_train_loss",
        "per_epoch_val_model_loss",
        "per_epoch_top1",
        "per_epoch_learning_rate",
        "replay_exposure_counts",
        "optimizer_step_counts",
        "final_raw_val_op_predictions",
        "final_calibrated_val_op_predictions",
        "operational_threshold_sweep",
        "initial_best_last_checkpoints",
        "last_checkpoint_optimizer_state",
        "training_seed",
        "machine_id",
        "resume_metadata",
    }
)


@dataclass(frozen=True)
class LiteratureRecord:
    evidence_id: str
    topic: str
    citation: str
    year: int
    primary_url: str
    doi: str
    stage1_hypothesis: str
    required_capabilities: tuple[str, ...]
    supporting_capabilities: tuple[str, ...]
    claim_boundary: str


_RECORDS = (
    LiteratureRecord(
        "MEMORIZATION_DYNAMICS",
        "memorization",
        "Arpit et al., A Closer Look at Memorization in Deep Networks, ICML 2017.",
        2017,
        "https://proceedings.mlr.press/v70/arpit17a.html",
        "",
        "Useful replay should be learnable before the network enters a late memorization regime.",
        ("per_sample_oof_probability_trajectory", "sample_labels"),
        ("per_epoch_train_loss", "per_epoch_val_model_loss"),
        "The paper motivates a dynamics hypothesis; it does not prove that Stage1 replay causes the same mechanism.",
    ),
    LiteratureRecord(
        "FORGETTING_EVENTS",
        "forgetting",
        "Toneva et al., An Empirical Study of Example Forgetting during Deep Neural Network Learning, ICLR 2019.",
        2019,
        "https://openreview.net/forum?id=BJlxm30cKm",
        "",
        "Frequently forgotten replay samples should be less reliable than samples learned and retained.",
        ("per_sample_oof_probability_trajectory", "sample_labels"),
        ("selection_training_dynamic_summaries",),
        "Stage1 can reproduce threshold-defined forgetting counts, but operational-threshold forgetting is a new definition.",
    ),
    LiteratureRecord(
        "DATASET_CARTOGRAPHY",
        "dataset_cartography",
        "Swayamdipta et al., Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics, EMNLP 2020.",
        2020,
        "https://aclanthology.org/2020.emnlp-main.746/",
        "10.18653/v1/2020.emnlp-main.746",
        "Confidence and variability trajectories should separate learnable boundary samples from persistently ambiguous samples.",
        ("per_sample_oof_probability_trajectory", "sample_labels"),
        ("selection_training_dynamic_summaries",),
        "The Stage1 binary OOF setting differs from the original NLP experiments, so bucket names are diagnostic rather than universal.",
    ),
    LiteratureRecord(
        "AUM_MARGIN",
        "aum",
        "Pleiss et al., Identifying Mislabeled Data using the Area Under the Margin Ranking, NeurIPS 2020.",
        2020,
        "https://proceedings.neurips.cc/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html",
        "",
        "Low or persistently negative target margins may identify noise-like replay candidates that should be excluded.",
        ("binary_logit_margin_reconstructable", "sample_labels"),
        ("per_sample_oof_probability_trajectory",),
        "Only binary margins reconstructed from saved probabilities are testable; multiclass logit geometry was not collected.",
    ),
    LiteratureRecord(
        "RHO_LOSS",
        "rho_loss",
        "Mindermann et al., Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt, ICML 2022.",
        2022,
        "https://proceedings.mlr.press/v162/mindermann22a.html",
        "",
        "Replay value may depend on reducible loss rather than final confidence or raw difficulty.",
        ("per_sample_holdout_loss_trajectory", "independent_irreducible_loss_model"),
        ("per_sample_oof_probability_trajectory",),
        "True RHO-LOSS is not recoverable from OOF probabilities because the independent irreducible-loss estimate is absent.",
    ),
    LiteratureRecord(
        "EL2N_ERROR_NORM",
        "grand_el2n",
        "Paul, Ganguli, and Dziugaite, Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021.",
        2021,
        "https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html",
        "",
        "Binary EL2N reconstructed from early OOF probabilities may describe difficulty, but need not imply beneficial replay direction.",
        ("per_sample_oof_probability_trajectory", "sample_labels"),
        ("selection_training_dynamic_summaries",),
        "Binary probability trajectories permit EL2N reconstruction; this does not validate GraNd or a causal value score.",
    ),
    LiteratureRecord(
        "GRAND_MAGNITUDE",
        "grand_el2n",
        "Paul, Ganguli, and Dziugaite, Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021.",
        2021,
        "https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html",
        "",
        "Per-example gradient norms may distinguish influential from inert replay samples.",
        ("per_sample_gradient",),
        ("per_sample_oof_probability_trajectory",),
        "GraNd is not testable because no per-sample gradients were collected; loss or probability changes are not gradient norms.",
    ),
    LiteratureRecord(
        "INFLUENCE_ALIGNMENT",
        "influence",
        "Koh and Liang, Understanding Black-box Predictions via Influence Functions, ICML 2017.",
        2017,
        "https://proceedings.mlr.press/v70/koh17a.html",
        "",
        "A replay sample is useful only when its parameter effect improves the protected operational tail.",
        ("per_sample_gradient", "inverse_hessian_vector_product"),
        ("final_raw_val_op_predictions",),
        "Influence functions are not testable from predictions alone; gradients and inverse-Hessian products are absent.",
    ),
    LiteratureRecord(
        "TRACIN_ALIGNMENT",
        "tracin",
        "Pruthi et al., Estimating Training Data Influence by Tracing Gradient Descent, NeurIPS 2020.",
        2020,
        "https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html",
        "",
        "Checkpoint-wise gradient alignment could separate replay samples that help or harm weak defects.",
        ("per_sample_gradient", "intermediate_checkpoints"),
        ("initial_best_last_checkpoints",),
        "TracIn is not testable because per-sample gradients and intermediate training checkpoints were not retained.",
    ),
    LiteratureRecord(
        "GRAD_MATCH",
        "gradient_matching",
        "Killamsetty et al., GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training, ICML 2021.",
        2021,
        "https://proceedings.mlr.press/v139/killamsetty21a.html",
        "",
        "A replay subset whose aggregate gradient aligns with the protected-tail objective should outperform confidence-only selection.",
        ("per_sample_gradient_embedding", "target_set_gradient"),
        ("final_raw_val_op_predictions",),
        "Gradient matching is not testable because neither candidate gradient embeddings nor target-tail gradients were collected.",
    ),
    LiteratureRecord(
        "REPLAY_STABILITY",
        "replay_overfit",
        "Rolnick et al., Experience Replay for Continual Learning, NeurIPS 2019.",
        2019,
        "https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html",
        "",
        "Replay can preserve useful behavior, but its dose and composition must be compared at equal exposure and seed.",
        ("replay_exposure_counts", "training_seed", "final_raw_val_op_predictions"),
        ("optimizer_step_counts",),
        "The paper studies continual learning; it supports replay controls but does not establish the Stage1 ranking mechanism.",
    ),
    LiteratureRecord(
        "REPLAY_LOGIT_CONSISTENCY",
        "replay_overfit",
        "Buzzega et al., Dark Experience for General Continual Learning: a Strong, Simple Baseline, NeurIPS 2020.",
        2020,
        "https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html",
        "",
        "Late replay overfit may be reduced by protecting prior function values rather than repeatedly forcing hard labels.",
        ("per_epoch_val_op_predictions", "stored_replay_logits"),
        ("per_epoch_train_loss", "final_raw_val_op_predictions"),
        "Stage1 did not collect per-epoch operational predictions or stored replay logits, so this is a future intervention hypothesis.",
    ),
    LiteratureRecord(
        "SGD_ALGORITHMIC_STABILITY",
        "learning_rate_stability",
        "Hardt, Recht, and Singer, Train faster, generalize better: Stability of stochastic gradient descent, ICML 2016.",
        2016,
        "https://proceedings.mlr.press/v48/hardt16.html",
        "",
        "Longer replay exposure and learning-rate-weighted updates may increase instability and degrade held-out operational performance.",
        ("per_epoch_learning_rate", "optimizer_step_counts", "per_epoch_val_model_loss"),
        ("replay_exposure_counts",),
        "The stability theorem assumptions do not hold exactly for YOLO/MuSGD; Stage1 tests associations and interactions only.",
    ),
    LiteratureRecord(
        "EDGE_OF_STABILITY",
        "edge_of_stability",
        "Cohen et al., Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability, ICLR 2021.",
        2021,
        "https://openreview.net/forum?id=jh-rTtvkGeM",
        "",
        "Late loss oscillation may interact with the learning-rate schedule and replay composition near an instability boundary.",
        ("per_epoch_learning_rate", "per_epoch_train_loss", "hessian_largest_eigenvalue"),
        ("per_epoch_val_model_loss",),
        "Loss and learning rate are observed, but sharpness is not; therefore edge-of-stability itself cannot be diagnosed.",
    ),
    LiteratureRecord(
        "PARAMETER_DRIFT",
        "parameter_drift",
        "Frankle et al., Linear Mode Connectivity and the Lottery Ticket Hypothesis, ICML 2020.",
        2020,
        "https://proceedings.mlr.press/v119/frankle20a.html",
        "",
        "Layer-wise parameter displacement may reveal whether harmful replay produces a different late optimization path.",
        ("initial_best_last_checkpoints",),
        ("training_seed", "machine_id"),
        "Endpoint weight displacement is directly measurable, but mode connectivity requires interpolation evaluations not present here.",
    ),
    LiteratureRecord(
        "SWA_WEIGHT_AVERAGING",
        "swa_mode_connectivity",
        "Izmailov et al., Averaging Weights Leads to Wider Optima and Better Generalization, UAI 2018.",
        2018,
        "https://arxiv.org/abs/1803.05407",
        "",
        "Averaging late checkpoints might mitigate the harmful late replay trajectory observed in some Stage1 runs.",
        ("intermediate_checkpoints", "checkpoint_interpolation_evaluations"),
        ("initial_best_last_checkpoints",),
        "Only initial, best, and last checkpoints exist; SWA or flatness cannot be retrospectively evaluated as defined.",
    ),
    LiteratureRecord(
        "MODE_CONNECTIVITY",
        "swa_mode_connectivity",
        "Garipov et al., Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs, NeurIPS 2018.",
        2018,
        "https://proceedings.neurips.cc/paper/2018/hash/be3087e74e9100d4bc4c6268cdbe8456-Abstract.html",
        "",
        "Good and harmful seed outcomes may occupy different basins or paths despite sharing a replay selection.",
        ("checkpoint_interpolation_evaluations", "intermediate_checkpoints"),
        ("initial_best_last_checkpoints",),
        "Weight endpoints alone do not establish connectivity, basin flatness, or an ensembling benefit.",
    ),
    LiteratureRecord(
        "VALIDATION_EARLY_STOPPING",
        "early_stopping",
        "Prechelt, Early Stopping - but when?, Neural Networks: Tricks of the Trade, 1998.",
        1998,
        "https://doi.org/10.1007/3-540-49430-8_3",
        "10.1007/3-540-49430-8_3",
        "Operational performance may peak before top-1 accuracy, so checkpoint rules must be validated on the intended constraint.",
        ("per_epoch_val_model_loss", "per_epoch_top1"),
        ("per_epoch_val_op_predictions",),
        "Stage1 can compare validation-curve stopping proxies, but cannot recover per-epoch operational TN/FN without saved predictions.",
    ),
    LiteratureRecord(
        "OPERATIONAL_NEYMAN_PEARSON",
        "neyman_pearson",
        "Scott and Nowak, A Neyman-Pearson Approach to Statistical Learning, IEEE TIT 2005.",
        2005,
        "https://doi.org/10.1109/TIT.2005.856955",
        "10.1109/TIT.2005.856955",
        "Model comparisons should maximize normal rejection while explicitly constraining defect misses, not slide an unconstrained threshold.",
        ("final_raw_val_op_predictions", "sample_labels", "operational_threshold_sweep"),
        ("final_calibrated_val_op_predictions",),
        "The finite validation set supports an operational constrained comparison, not a distribution-free deployment guarantee.",
    ),
    LiteratureRecord(
        "PARTIAL_AUC_TAIL_ORDERING",
        "partial_auc",
        "Narasimhan and Agarwal, A Structural SVM Based Approach for Optimizing Partial AUC, ICML 2013.",
        2013,
        "https://proceedings.mlr.press/v28/narasimhan13.html",
        "",
        "Restricted ROC performance near the deployed high-recall region should distinguish genuine ranking gains from threshold movement.",
        ("final_raw_val_op_predictions", "sample_labels"),
        ("operational_threshold_sweep",),
        "Stage1 evaluates partial-AUC/frontier diagnostics after training; it did not optimize the paper's structural surrogate.",
    ),
    LiteratureRecord(
        "CALIBRATION_VS_RANKING",
        "neyman_pearson",
        "Niculescu-Mizil and Caruana, Predicting Good Probabilities with Supervised Learning, ICML 2005.",
        2005,
        "https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf",
        "",
        "A monotone calibration map may change probability interpretation but cannot repair a damaged raw ranking frontier.",
        ("final_raw_val_op_predictions", "final_calibrated_val_op_predictions"),
        ("operational_threshold_sweep",),
        "This supports a raw-versus-calibrated diagnostic; it does not validate Platt calibration on an external population.",
    ),
)


def classify_testability(
    required_capabilities: Iterable[str],
    collected_capabilities: Iterable[str],
) -> str:
    """Classify whether the exact literature-motivated quantity was collected."""

    required = frozenset(required_capabilities)
    collected = frozenset(collected_capabilities)
    if not required:
        return CONTEXT_ONLY
    present = required & collected
    if present == required:
        return DIRECTLY_TESTABLE
    if present:
        return PARTIALLY_TESTABLE
    return NOT_TESTABLE


def build_literature_evidence_matrix(
    collected_capabilities: Iterable[str] = DEFAULT_COLLECTED_CAPABILITIES,
) -> pd.DataFrame:
    """Return the primary-source registry with explicit Stage1 testability."""

    collected = frozenset(collected_capabilities)
    rows = []
    for record in _RECORDS:
        row = asdict(record)
        required = frozenset(record.required_capabilities)
        row["required_capabilities"] = ";".join(record.required_capabilities)
        row["supporting_capabilities"] = ";".join(record.supporting_capabilities)
        row["present_required_capabilities"] = ";".join(sorted(required & collected))
        row["missing_required_capabilities"] = ";".join(sorted(required - collected))
        row["testability_status"] = classify_testability(required, collected)
        rows.append(row)
    matrix = pd.DataFrame(rows)
    assert_literature_matrix_integrity(matrix)
    return matrix


def assert_literature_matrix_integrity(matrix: pd.DataFrame) -> None:
    """Reject duplicate, incomplete, or search-result-based citations."""

    required_columns = {
        "evidence_id",
        "topic",
        "citation",
        "year",
        "primary_url",
        "doi",
        "stage1_hypothesis",
        "required_capabilities",
        "supporting_capabilities",
        "claim_boundary",
        "present_required_capabilities",
        "missing_required_capabilities",
        "testability_status",
    }
    missing = sorted(required_columns - set(matrix.columns))
    if missing:
        raise LiteratureEvidenceError(f"missing columns: {missing}")
    duplicates = matrix.loc[matrix["evidence_id"].duplicated(), "evidence_id"].tolist()
    if duplicates:
        raise LiteratureEvidenceError(f"duplicate evidence_id: {duplicates}")
    allowed_statuses = {
        DIRECTLY_TESTABLE,
        PARTIALLY_TESTABLE,
        NOT_TESTABLE,
        CONTEXT_ONLY,
    }
    bad_status = sorted(set(matrix["testability_status"]) - allowed_statuses)
    if bad_status:
        raise LiteratureEvidenceError(f"invalid testability statuses: {bad_status}")
    for row in matrix.itertuples(index=False):
        parsed = urlparse(str(row.primary_url))
        host = parsed.netloc.lower()
        if parsed.scheme != "https" or not host or "google." in host or "/search" in parsed.path:
            raise LiteratureEvidenceError(
                f"not a primary source URL for {row.evidence_id}: {row.primary_url}"
            )
        if not str(row.citation).strip() or not str(row.stage1_hypothesis).strip():
            raise LiteratureEvidenceError(f"incomplete evidence row: {row.evidence_id}")


__all__ = [
    "CONTEXT_ONLY",
    "DEFAULT_COLLECTED_CAPABILITIES",
    "DIRECTLY_TESTABLE",
    "LiteratureEvidenceError",
    "NOT_TESTABLE",
    "PARTIALLY_TESTABLE",
    "assert_literature_matrix_integrity",
    "build_literature_evidence_matrix",
    "classify_testability",
]
