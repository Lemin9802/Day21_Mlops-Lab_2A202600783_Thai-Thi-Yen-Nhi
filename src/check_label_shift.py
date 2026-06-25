import json
import os

import pandas as pd


TRAIN_PATH = "data/train_phase1.csv"
EVAL_PATH = "data/eval.csv"
LABEL_COL = "target"

OUTPUT_JSON = "outputs/label_shift_metrics.json"
OUTPUT_REPORT = "outputs/label_shift_report.md"

SHIFT_THRESHOLD = 0.10


def label_distribution(df: pd.DataFrame) -> dict[str, float]:
    """
    Tính phân phối label theo tỷ lệ phần trăm.

    Ví dụ:
        label 0 chiếm 0.30
        label 1 chiếm 0.50
        label 2 chiếm 0.20
    """
    counts = df[LABEL_COL].value_counts(normalize=True).sort_index()
    return {str(label): float(value) for label, value in counts.items()}


def main() -> None:
    """
    So sánh label distribution giữa train và eval.

    Nếu độ lệch lớn nhất vượt SHIFT_THRESHOLD thì tạo cảnh báo.
    Script không fail pipeline; nó tạo report để reviewer thấy drift/shift risk.
    """
    os.makedirs("outputs", exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    train_dist = label_distribution(train_df)
    eval_dist = label_distribution(eval_df)

    all_labels = sorted(set(train_dist) | set(eval_dist))

    deltas = {}
    for label in all_labels:
        train_value = train_dist.get(label, 0.0)
        eval_value = eval_dist.get(label, 0.0)
        deltas[label] = abs(eval_value - train_value)

    max_shift = max(deltas.values()) if deltas else 0.0
    shift_detected = max_shift > SHIFT_THRESHOLD

    metrics = {
        "shift_threshold": SHIFT_THRESHOLD,
        "max_label_shift": max_shift,
        "shift_detected": shift_detected,
        "train_distribution": train_dist,
        "eval_distribution": eval_dist,
        "absolute_deltas": deltas,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_lines = [
        "# Label Distribution Shift Report",
        "",
        f"- Shift threshold: {SHIFT_THRESHOLD:.2f}",
        f"- Max label shift: {max_shift:.4f}",
        f"- Shift detected: {shift_detected}",
        "",
        "## Distribution comparison",
        "",
        "| Label | Train distribution | Eval distribution | Absolute delta |",
        "|---|---:|---:|---:|",
    ]

    for label in all_labels:
        report_lines.append(
            "| "
            f"{label} | "
            f"{train_dist.get(label, 0.0):.4f} | "
            f"{eval_dist.get(label, 0.0):.4f} | "
            f"{deltas[label]:.4f} |"
        )

    report_lines.append("")

    if shift_detected:
        report_lines.append(
            "⚠️ Warning: label distribution shift exceeds the configured threshold."
        )
    else:
        report_lines.append(
            "No significant label distribution shift detected."
        )

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Max label shift: {max_shift:.4f}")
    print(f"Shift threshold: {SHIFT_THRESHOLD:.4f}")
    print(f"Shift detected: {shift_detected}")
    print(f"Report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()