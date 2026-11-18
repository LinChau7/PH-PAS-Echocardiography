"""Patient-level multi-video probability aggregation for PH-PAS.

This script is a post-processing utility. It does not load a neural-network
checkpoint or process videos. It combines video-level Low/Intermediate/High
probabilities for all eligible videos belonging to the same index TTE
examination.

Weighting rule (matching the study implementation):
    confidence_i = max(p_low_i, p_intermediate_i, p_high_i)
    weight_i = confidence_i * view_accuracy[view_i] * modality_accuracy[modality_i]

For each index examination, the patient-level probability distribution is the
normalized weighted average of the video-level probability distributions.
If the total weight is zero, an unweighted arithmetic mean is used as a
numerical fallback, matching the previous implementation. A single eligible
video is passed through unchanged.

View and modality accuracies must be estimated from video-level predictions
produced by the final selected checkpoint on the dev subset of the
model-development training cohort. Save the resulting weights to JSON and
keep the same weights fixed for the internal-test and external-validation
cohorts. This script performs weight estimation and aggregation; it does not
run checkpoint inference itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
CLASS_NAMES = ["Low", "Intermediate", "High"]
DEFAULT_PROB_COLUMNS = ["prob_low", "prob_intermediate", "prob_high"]
TRUE_VALUES = {"1", "1.0", "true", "t", "yes", "y", "pass", "passed", "eligible", "include", "included"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n", "fail", "failed", "ineligible", "exclude", "excluded"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV column(s): {missing}")


def _clean_category(series: pd.Series, column_name: str) -> pd.Series:
    if series.isna().any():
        rows = series.index[series.isna()].tolist()[:10]
        raise ValueError(
            f"Column '{column_name}' contains missing values (example row indices: {rows})."
        )
    values = series.astype(str).str.strip()
    if (values == "").any():
        raise ValueError(f"Column '{column_name}' contains empty category labels.")
    return values


def _parse_qc_values(series: pd.Series, column_name: str) -> pd.Series:
    """Parse explicit QC pass/fail values; unknown or missing values are errors."""
    parsed: list[bool] = []
    invalid: list[tuple[Any, Any]] = []
    for index, value in series.items():
        if pd.isna(value):
            invalid.append((index, value))
            continue
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value) == 1.0:
                parsed.append(True)
            elif float(value) == 0.0:
                parsed.append(False)
            else:
                invalid.append((index, value))
            continue
        text = str(value).strip().lower()
        if text in TRUE_VALUES:
            parsed.append(True)
        elif text in FALSE_VALUES:
            parsed.append(False)
        else:
            invalid.append((index, value))
    if invalid:
        examples = invalid[:10]
        raise ValueError(
            f"Column '{column_name}' contains missing or unrecognized QC values; "
            f"use explicit pass/fail values such as 1/0 or true/false. Examples: {examples}"
        )
    return pd.Series(parsed, index=series.index, dtype=bool)


def _get_probabilities(
    df: pd.DataFrame,
    probability_columns: list[str],
    sum_tolerance: float,
) -> np.ndarray:
    _require_columns(df, probability_columns)
    try:
        probabilities = df[probability_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError("The three probability columns must contain numeric values.") from exc

    if not np.isfinite(probabilities).all():
        raise ValueError("Video-level probabilities contain NaN or infinite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Video-level probabilities must be between 0 and 1 (not percentages).")

    row_sums = probabilities.sum(axis=1)
    bad = np.abs(row_sums - 1.0) > sum_tolerance
    if bad.any():
        examples = [(int(i), float(row_sums[i])) for i in np.flatnonzero(bad)[:10]]
        raise ValueError(
            "Each row's Low/Intermediate/High probabilities must sum to 1 "
            f"(tolerance={sum_tolerance}). Example row indices and sums: {examples}"
        )
    return probabilities


def _apply_qc_filter(
    df: pd.DataFrame,
    qc_column: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if qc_column is None:
        return df.copy(), pd.Series(True, index=df.index, dtype=bool)
    _require_columns(df, [qc_column])
    qc_pass = _parse_qc_values(df[qc_column], qc_column)
    return df.loc[qc_pass].copy(), qc_pass


def fit_performance_weights(
    input_csv: str | Path,
    output_json: str | Path,
    *,
    label_column: str = "PH_label",
    view_column: str = "View",
    modality_column: str = "Modality",
    probability_columns: list[str] | None = None,
    qc_column: str | None = None,
    probability_sum_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Estimate view/modality accuracy weights from final-checkpoint dev predictions."""
    probability_columns = probability_columns or DEFAULT_PROB_COLUMNS
    df = _read_csv(input_csv)
    _require_columns(df, [label_column, view_column, modality_column, *probability_columns])

    df, _ = _apply_qc_filter(df, qc_column)
    if df.empty:
        raise ValueError("No QC-eligible rows remain for estimating performance weights.")

    probabilities = _get_probabilities(df, probability_columns, probability_sum_tolerance)
    try:
        label_values = pd.to_numeric(df[label_column], errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"Column '{label_column}' must contain integer class labels 0, 1, or 2.") from exc
    if (not np.isfinite(label_values).all()) or not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError(f"Column '{label_column}' must contain finite integer class labels.")
    labels = label_values.astype(int)
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError(f"Column '{label_column}' must use 0=Low, 1=Intermediate, 2=High.")

    views = _clean_category(df[view_column], view_column)
    modalities = _clean_category(df[modality_column], modality_column)
    predicted = probabilities.argmax(axis=1)

    view_weights: dict[str, float] = {}
    view_n: dict[str, int] = {}
    for view in sorted(views.unique()):
        mask = (views == view).to_numpy()
        view_weights[str(view)] = float(np.mean(predicted[mask] == labels[mask]))
        view_n[str(view)] = int(mask.sum())

    modality_weights: dict[str, float] = {}
    modality_n: dict[str, int] = {}
    for modality in sorted(modalities.unique()):
        mask = (modalities == modality).to_numpy()
        modality_weights[str(modality)] = float(np.mean(predicted[mask] == labels[mask]))
        modality_n[str(modality)] = int(mask.sum())

    payload: dict[str, Any] = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "weighting_rule": "max_class_probability * view_accuracy * modality_accuracy",
        "confidence_definition": "maximum of the Low, Intermediate, and High video-level probabilities",
        "performance_metric": "video-level accuracy",
        "weight_source": (
            "dev subset of the model-development training cohort; input video-level "
            "predictions are expected to come from the final selected checkpoint"
        ),
        "qc_filter_applied": qc_column is not None,
        "qc_column": qc_column,
        "classes": {"0": "Low", "1": "Intermediate", "2": "High"},
        "probability_columns": probability_columns,
        "view_weights": view_weights,
        "view_sample_sizes": view_n,
        "modality_weights": modality_weights,
        "modality_sample_sizes": modality_n,
        "n_video_rows_used": int(len(df)),
        "note": (
            "Keep these weights fixed. Do not re-estimate them using internal-test "
            "or external-validation labels. Category keys are stored as strings "
            "and must match the corresponding input CSV labels exactly."
        ),
    }

    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _load_weights(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Weight JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Weight JSON must contain a JSON object.")
    for key in ("view_weights", "modality_weights"):
        if key not in payload or not isinstance(payload[key], dict) or not payload[key]:
            raise ValueError(f"Weight JSON must contain a non-empty '{key}' object.")
        for category, value in payload[key].items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Weight '{key}.{category}' must be numeric.") from exc
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"Weight '{key}.{category}' must be finite and non-negative.")
    return payload


def aggregate_video_predictions(
    input_csv: str | Path,
    weights_json: str | Path,
    output_csv: str | Path,
    *,
    group_column: str = "report_id",
    label_column: str = "PH_label",
    view_column: str = "View",
    modality_column: str = "Modality",
    probability_columns: list[str] | None = None,
    qc_column: str | None = None,
    probability_sum_tolerance: float = 1e-3,
    audit_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate eligible video-level probabilities within each index TTE examination."""
    probability_columns = probability_columns or DEFAULT_PROB_COLUMNS
    df = _read_csv(input_csv)
    _require_columns(df, [group_column, label_column, view_column, modality_column, *probability_columns])
    if df[group_column].isna().any():
        raise ValueError(f"Grouping column '{group_column}' contains missing values.")

    weights = _load_weights(weights_json)

    try:
        label_values = pd.to_numeric(df[label_column], errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"Column '{label_column}' must contain integer class labels 0, 1, or 2.") from exc
    if (not np.isfinite(label_values).all()) or not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError(f"Column '{label_column}' must contain finite integer class labels.")
    labels_all = label_values.astype(int)
    if not np.isin(labels_all, [0, 1, 2]).all():
        raise ValueError(f"Column '{label_column}' must use 0=Low, 1=Intermediate, 2=High.")

    if qc_column is not None:
        _require_columns(df, [qc_column])
        qc_pass = _parse_qc_values(df[qc_column], qc_column)
    else:
        qc_pass = pd.Series(True, index=df.index, dtype=bool)

    # Validate predictions and category labels only for QC-eligible videos.
    # Excluded videos may legitimately have no model prediction or metadata.
    probabilities_all = np.full((len(df), 3), np.nan, dtype=float)
    eligible_positions = np.flatnonzero(qc_pass.to_numpy())
    if len(eligible_positions) > 0:
        eligible_df = df.iloc[eligible_positions]
        probabilities_all[eligible_positions] = _get_probabilities(
            eligible_df, probability_columns, probability_sum_tolerance
        )
        eligible_views = _clean_category(eligible_df[view_column], view_column)
        eligible_modalities = _clean_category(eligible_df[modality_column], modality_column)
    else:
        eligible_views = pd.Series(dtype=str)
        eligible_modalities = pd.Series(dtype=str)

    # Resolve all view/modality weights before aggregation; unknown categories
    # must not silently receive zero weights.
    missing_views = sorted(set(eligible_views) - set(weights["view_weights"]))
    missing_modalities = sorted(set(eligible_modalities) - set(weights["modality_weights"]))
    if missing_views or missing_modalities:
        raise ValueError(
            "The aggregation input contains view/modality categories absent from the "
            "development-derived weights. Revisit the development weights; do not "
            f"guess values. Missing views={missing_views}; missing modalities={missing_modalities}."
        )

    views_all = pd.Series(index=df.index, dtype=object)
    modalities_all = pd.Series(index=df.index, dtype=object)
    views_all.iloc[eligible_positions] = eligible_views.to_numpy()
    modalities_all.iloc[eligible_positions] = eligible_modalities.to_numpy()

    view_weight_all = np.full(len(df), np.nan, dtype=float)
    modality_weight_all = np.full(len(df), np.nan, dtype=float)
    view_weight_all[eligible_positions] = np.array(
        [float(weights["view_weights"][value]) for value in eligible_views], dtype=float
    )
    modality_weight_all[eligible_positions] = np.array(
        [float(weights["modality_weights"][value]) for value in eligible_modalities], dtype=float
    )
    confidence_all = np.full(len(df), np.nan, dtype=float)
    confidence_all[eligible_positions] = probabilities_all[eligible_positions].max(axis=1)
    raw_weight_all = confidence_all * view_weight_all * modality_weight_all

    audit = df.copy()
    audit["qc_eligible"] = qc_pass.to_numpy()
    audit["prediction_confidence"] = confidence_all
    audit["view_performance_weight"] = view_weight_all
    audit["modality_performance_weight"] = modality_weight_all
    audit["video_aggregation_weight"] = raw_weight_all
    audit["normalized_aggregation_weight"] = np.nan
    audit["included_in_patient_aggregation"] = qc_pass.to_numpy()

    output_rows: list[dict[str, Any]] = []
    # groupby(sort=False) preserves the first-seen examination order.
    for group_value, group_indices in df.groupby(group_column, sort=False, dropna=False).groups.items():
        idx = np.asarray(list(group_indices), dtype=int)
        eligible_idx = idx[qc_pass.loc[idx].to_numpy()]
        n_input = int(len(idx))
        n_excluded = int(n_input - len(eligible_idx))
        n_eligible = int(len(eligible_idx))

        group_labels = np.unique(labels_all[idx])
        if len(group_labels) != 1:
            raise ValueError(
                f"Videos grouped under {group_column}={group_value!r} have inconsistent "
                f"reference labels in '{label_column}': {group_labels.tolist()}"
            )

        row: dict[str, Any] = {
            group_column: group_value,
            label_column: int(group_labels[0]),
            "n_videos_input": n_input,
            "n_videos_qc_excluded": n_excluded,
            "n_videos_aggregated": n_eligible,
        }

        if n_eligible == 0:
            row.update({
                "prob_low": np.nan,
                "prob_intermediate": np.nan,
                "prob_high": np.nan,
                "predicted_class_index": np.nan,
                "predicted_class": "",
                "aggregation_status": "no_qc_eligible_videos",
            })
            output_rows.append(row)
            continue

        group_probs = probabilities_all[eligible_idx]
        if n_eligible == 1:
            aggregated = group_probs[0].copy()
            normalized_weights = np.array([1.0])
            status = "single_video_direct"
        else:
            group_weights = raw_weight_all[eligible_idx]
            weight_sum = float(group_weights.sum())
            if weight_sum > 0:
                normalized_weights = group_weights / weight_sum
                aggregated = np.sum(group_probs * normalized_weights[:, None], axis=0)
                status = "confidence_view_modality_weighted"
            else:
                normalized_weights = np.full(n_eligible, 1.0 / n_eligible)
                aggregated = group_probs.mean(axis=0)
                status = "unweighted_fallback_zero_total_weight"

        # Protect against tiny floating-point drift while retaining the computed
        # weighted-average proportions.
        aggregated = aggregated / aggregated.sum()
        predicted_class_index = int(np.argmax(aggregated))
        row.update({
            "prob_low": float(aggregated[0]),
            "prob_intermediate": float(aggregated[1]),
            "prob_high": float(aggregated[2]),
            "predicted_class_index": predicted_class_index,
            "predicted_class": CLASS_NAMES[predicted_class_index],
            "aggregation_status": status,
        })
        audit.loc[eligible_idx, "normalized_aggregation_weight"] = normalized_weights
        output_rows.append(row)

    output_df = pd.DataFrame(output_rows)
    output_path = Path(output_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if audit_csv is None:
        audit_path = output_path.with_name(output_path.stem + "_video_audit.csv")
    else:
        audit_path = Path(audit_csv).expanduser()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    return output_df


def _add_common_probability_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--view-col", default="View", help="CSV column containing the view label (default: View).")
    parser.add_argument("--modality-col", default="Modality", help="CSV column containing the modality label (default: Modality).")
    parser.add_argument("--prob-low-col", default="prob_low")
    parser.add_argument("--prob-intermediate-col", default="prob_intermediate")
    parser.add_argument("--prob-high-col", default="prob_high")
    parser.add_argument(
        "--qc-col",
        default=None,
        help="Optional QC pass/fail column. If omitted, input rows are assumed to have been QC-filtered already.",
    )
    parser.add_argument(
        "--probability-sum-tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute deviation of each probability row sum from 1 (default: 0.001).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit development-set view/modality weights or aggregate PH-PAS video probabilities at index-examination level."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit-weights",
        help="Estimate view/modality accuracy weights from final-checkpoint predictions on the development dev subset only.",
    )
    fit_parser.add_argument("--input-csv", required=True, help="Video-level predictions from the dev subset, generated using the final selected checkpoint.")
    fit_parser.add_argument("--output-json", required=True, help="Path for the fixed aggregation-weight JSON.")
    fit_parser.add_argument("--label-col", default="PH_label", help="True video-level class label column (0/1/2).")
    _add_common_probability_args(fit_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate video-level predictions using previously fixed development-set weights.",
    )
    aggregate_parser.add_argument("--input-csv", required=True, help="Video-level predictions for one evaluation cohort.")
    aggregate_parser.add_argument("--weights-json", required=True, help="JSON produced by fit-weights using development data.")
    aggregate_parser.add_argument("--output-csv", required=True, help="Patient/index-examination-level output CSV.")
    aggregate_parser.add_argument(
        "--label-col",
        default="PH_label",
        help="Reference class label column; videos within an index examination must share one label (default: PH_label).",
    )
    aggregate_parser.add_argument(
        "--group-col",
        default="report_id",
        help="Unique index TTE examination identifier (default: report_id; Chinese source column can be passed as --group-col 报告ID).",
    )
    aggregate_parser.add_argument("--audit-csv", default=None, help="Optional path for per-video weight/QC audit CSV.")
    _add_common_probability_args(aggregate_parser)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probability_columns = [args.prob_low_col, args.prob_intermediate_col, args.prob_high_col]

    if args.command == "fit-weights":
        payload = fit_performance_weights(
            input_csv=args.input_csv,
            output_json=args.output_json,
            label_column=args.label_col,
            view_column=args.view_col,
            modality_column=args.modality_col,
            probability_columns=probability_columns,
            qc_column=args.qc_col,
            probability_sum_tolerance=args.probability_sum_tolerance,
        )
        print(f"Saved fixed development-set aggregation weights: {args.output_json}")
        print(f"Video rows used: {payload['n_video_rows_used']}")
        print(f"View weights (accuracy): {payload['view_weights']}")
        print(f"Modality weights (accuracy): {payload['modality_weights']}")
        return

    result = aggregate_video_predictions(
        input_csv=args.input_csv,
        weights_json=args.weights_json,
        output_csv=args.output_csv,
        group_column=args.group_col,
        label_column=args.label_col,
        view_column=args.view_col,
        modality_column=args.modality_col,
        probability_columns=probability_columns,
        qc_column=args.qc_col,
        probability_sum_tolerance=args.probability_sum_tolerance,
        audit_csv=args.audit_csv,
    )
    print(f"Saved index-examination-level predictions: {args.output_csv}")
    print(f"Saved per-video audit: {args.audit_csv or str(Path(args.output_csv).with_name(Path(args.output_csv).stem + '_video_audit.csv'))}")
    print(f"Index examinations processed: {len(result)}")
    if "aggregation_status" in result:
        print("Aggregation status counts:")
        print(result["aggregation_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
