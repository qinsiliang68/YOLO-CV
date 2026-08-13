from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.evaluation import compute_tie_safe_frontier, validate_checkpoint_for_evaluation, write_frontier_artifacts
from stage1_sctsr_v4.prediction_artifact import read_registered_prediction_artifact
from stage1_sctsr_v4.serialization import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one registered prediction/checkpoint identity with the tie-safe raw frontier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--mode", choices=["formal", "trajectory", "synthetic"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-summary", type=Path)
    parser.add_argument("--frontier-output", type=Path, required=True)
    parser.add_argument("--frontier-summary-output", type=Path)
    parser.add_argument("--max-fn", type=int, default=95)
    parser.add_argument("--target-tn", type=int, default=68_253)
    parser.add_argument("--allow-synthetic-columnar-fallback", action="store_true")
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.predictions.suffix.lower() != ".parquet":
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Raw JSON/CSV predictions cannot enter canonical evaluation")
        prediction_summary = arguments.prediction_summary or arguments.predictions.with_name("prediction_summary.json")
        frontier_summary = arguments.frontier_summary_output or arguments.frontier_output.with_name("frontier_summary.json")
        if arguments.frontier_output.suffix.lower() != ".parquet":
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Canonical frontier output must be a partitioned Parquet file")
        if arguments.mode == "formal" and (arguments.max_fn != 95 or arguments.target_tn != 68_253):
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Formal FN/TN anchors are frozen at FN=0..95 and TN=68253")
        rows, prediction_identity, binding = read_registered_prediction_artifact(
            arguments.predictions,
            summary_path=prediction_summary,
            checkpoint_path=arguments.checkpoint,
            evaluation_mode=arguments.mode,
            allow_synthetic_portable_fallback=arguments.allow_synthetic_columnar_fallback,
        )
        checkpoint = validate_checkpoint_for_evaluation(
            arguments.checkpoint,
            epoch=arguments.checkpoint_epoch,
            mode=arguments.mode,
            expected_sha256=binding.checkpoint_sha256,
            expected_source_tree_digest=binding.source_tree_digest,
            expected_training_seed=binding.training_seed,
        )
        if binding.checkpoint_epoch != arguments.checkpoint_epoch:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction checkpoint epoch differs from the CLI checkpoint")
        points, summary = compute_tie_safe_frontier(
            rows,
            max_fn=arguments.max_fn,
            target_tn=arguments.target_tn,
            checkpoint_sha256=checkpoint["checkpoint_sha256"],
            prediction_artifact_sha256=sha256_file(arguments.predictions),
        )
        frontier_manifest, frontier_identity = write_frontier_artifacts(
            points,
            summary,
            frontier_path=arguments.frontier_output,
            summary_path=frontier_summary,
            evaluation_mode=arguments.mode,
            split_role=binding.split_role,
            checkpoint_epoch=arguments.checkpoint_epoch,
        )
        return {
            "frontier_output": arguments.frontier_output.resolve().as_posix(),
            "frontier_summary_output": frontier_summary.resolve().as_posix(),
            "frontier_sha256": frontier_manifest.sha256,
            "frontier_row_count": frontier_manifest.row_count,
            "checkpoint_identity": checkpoint,
            "prediction_identity": dict(prediction_identity),
            "frontier_identity": frontier_identity,
            "mode": arguments.mode,
            "scientific_state": "SYNTHETIC_NOT_SCIENTIFIC_RESULT" if arguments.mode == "synthetic" else "ENDPOINT_EVIDENCE_NOT_METHOD_EFFECTIVENESS_CLAIM",
        }

    return run_cli("evaluate_checkpoint", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
