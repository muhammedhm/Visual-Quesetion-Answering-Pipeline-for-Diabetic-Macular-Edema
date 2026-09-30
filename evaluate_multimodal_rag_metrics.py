"""
Standalone DeepEyeNet VQA / Multimodal RAG evaluator.

This script does not modify the DeepEyeNet project. It imports the existing
project modules at runtime and writes evaluation reports to a separate output
folder.

Example:
    python evaluate_multimodal_rag_metrics.py ^
      --project-dir "C:/Users/ISHHAQ/OneDrive/Desktop/FYP_Project/DeepEyeNet/deepeye_2_way" ^
      --qa-file "C:/Users/ISHHAQ/OneDrive/Desktop/FYP_Project/DeepEyeNet/dme_vqa/qa/trainqa.json" ^
      --image-dir "C:/Users/ISHHAQ/OneDrive/Desktop/FYP_Project/DeepEyeNet/dme_vqa/visual/train" ^
      --architecture both ^
      --sample-size 100 ^
      --top-k 5

Quick run from the deepeye_2_way folder:
    python evaluate_multimodal_rag_metrics.py --architecture both --sample-size 100

Outputs:
    metrics_summary.json
    predictions.csv
    confusion_matrix_<architecture>.csv
    per_class_<architecture>.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SUPPORTED_ARCHITECTURES = {"vqa", "rag", "both"}
YES_NO_LABELS = ["no", "yes"]
GRADE_LABELS = ["0", "1", "2"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_project_dir = script_dir
    default_dataset_root = script_dir.parent / "dme_vqa"
    default_qa_file = default_dataset_root / "qa" / "trainqa.json"
    default_image_dir = default_dataset_root / "visual" / "train"

    parser = argparse.ArgumentParser(description="Evaluate DeepEyeNet VQA and Multimodal RAG metrics.")
    parser.add_argument(
        "--project-dir",
        default=str(default_project_dir),
        help=f"Folder containing api_app.py/model_inference.py/rag_engine.py. Default: {default_project_dir}",
    )
    parser.add_argument(
        "--qa-file",
        default=str(default_qa_file),
        help=f"DME VQA JSON file, usually trainqa.json/testqa.json. Default: {default_qa_file}",
    )
    parser.add_argument(
        "--image-dir",
        default=str(default_image_dir),
        help=f"Folder containing fundus images. Default: {default_image_dir}",
    )
    parser.add_argument("--output-dir", default="./deepeye_eval_outputs", help="Where metrics files will be written")
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="both")
    parser.add_argument("--sample-size", type=int, default=0, help="0 means evaluate all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap rounds for mean/std estimates")
    parser.add_argument("--user-id", type=int, default=0, help="User id for rag_pipeline rate limiter")
    parser.add_argument("--skip-inside", action="store_true", help="Skip region-mask questions if masks are unavailable")
    args = parser.parse_args()
    missing = []
    if not Path(args.project_dir).exists():
        missing.append(f"--project-dir path not found: {args.project_dir}")
    if not Path(args.qa_file).exists():
        missing.append(f"--qa-file path not found: {args.qa_file}")
    if not Path(args.image_dir).exists():
        missing.append(f"--image-dir path not found: {args.image_dir}")
    if missing:
        parser.error(
            "\n".join(missing)
            + "\nPass the paths explicitly if your dataset is stored somewhere else."
        )
    return args


def add_project_to_path(project_dir: str) -> None:
    project = str(Path(project_dir).resolve())
    if project not in sys.path:
        sys.path.insert(0, project)


def configure_project_modules(modules: dict[str, Any], project_dir: str) -> None:
    """
    Make project-relative paths absolute for standalone evaluation.

    rag_engine.py stores INDEX_PATH/META_PATH as ./rag_index/... which depends
    on the caller's current working directory. When this evaluator is launched
    from the parent DeepEyeNet folder, RAG search otherwise looks in the wrong
    place and reports "RAG index not found" even though deepeye_2_way/rag_index
    exists.
    """
    project = Path(project_dir).resolve()
    modules["rag_engine"].INDEX_PATH = str(project / "rag_index" / "faiss.index")
    modules["rag_engine"].META_PATH = str(project / "rag_index" / "metadata.pkl")


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    for grade in GRADE_LABELS:
        if text == grade or f"grade {grade}" in text:
            return grade
    return text


def dataset_type_to_project_type(row: dict[str, Any]) -> str | None:
    qt = str(row.get("question_type", "")).lower()
    question = str(row.get("question", "")).lower()
    if qt == "whole":
        return "whole"
    if qt == "fovea":
        return "fovea"
    if qt == "grade":
        return "grade"
    if qt == "inside":
        if "optic" in question or "disc" in question or "disk" in question:
            return "inside_optic"
        return "inside_exudates"
    if qt in {"inside_exudates", "inside_optic"}:
        return qt
    return None


def load_dataset(path: str, sample_size: int, seed: int, skip_inside: bool) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    usable = []
    for row in rows:
        project_type = dataset_type_to_project_type(row)
        if not project_type:
            continue
        if skip_inside and project_type.startswith("inside"):
            continue
        row = dict(row)
        row["project_question_type"] = project_type
        row["gold_normalized"] = normalize_answer(row.get("answer"))
        usable.append(row)
    if sample_size and sample_size < len(usable):
        random.Random(seed).shuffle(usable)
        usable = usable[:sample_size]
    return usable


def resolve_image(image_dir: str, image_name: str) -> Path:
    direct = Path(image_dir) / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = Path(image_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for {image_name}")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        if truth in index and pred in index:
            matrix[index[truth]][index[pred]] += 1
    return matrix


def per_class_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, labels)
    rows = []
    total = sum(sum(row) for row in matrix)
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels)) if r != i)
        fn = sum(matrix[i][c] for c in range(len(labels)) if c != i)
        tn = total - tp - fp - fn
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "label": label,
                "support": sum(matrix[i]),
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1": f1,
            }
        )
    return rows


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return safe_div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))


def balanced_accuracy(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    rows = per_class_metrics(y_true, y_pred, labels)
    present = [row["recall_sensitivity"] for row in rows if row["support"] > 0]
    return statistics.mean(present) if present else 0.0


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    rows = per_class_metrics(y_true, y_pred, labels)
    present = [row["f1"] for row in rows if row["support"] > 0]
    return statistics.mean(present) if present else 0.0


def quadratic_weighted_kappa(y_true: list[str], y_pred: list[str], labels: list[str]) -> float | None:
    if not y_true:
        return None
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    for truth, pred in zip(y_true, y_pred):
        if truth in label_to_idx and pred in label_to_idx:
            observed[label_to_idx[truth]][label_to_idx[pred]] += 1.0
    total = sum(sum(row) for row in observed)
    if total == 0:
        return None
    hist_true = [sum(observed[i][j] for j in range(n)) for i in range(n)]
    hist_pred = [sum(observed[i][j] for i in range(n)) for j in range(n)]
    weighted_observed = 0.0
    weighted_expected = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / ((n - 1) ** 2)
            expected = hist_true[i] * hist_pred[j] / total
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected
    if weighted_expected == 0:
        return 1.0
    return 1.0 - weighted_observed / weighted_expected


def mcnemar_p_value(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> float | None:
    by_id_a = {row["eval_id"]: row for row in rows_a}
    by_id_b = {row["eval_id"]: row for row in rows_b}
    b = c = 0
    for key, a in by_id_a.items():
        if key not in by_id_b:
            continue
        a_ok = a["correct"]
        b_ok = by_id_b[key]["correct"]
        if a_ok and not b_ok:
            b += 1
        elif not a_ok and b_ok:
            c += 1
    n = b + c
    if n == 0:
        return None
    if n <= 100:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        return min(1.0, 2 * tail)
    statistic = ((abs(b - c) - 1) ** 2) / n
    return math.erfc(math.sqrt(statistic / 2))


def bootstrap_metric(values: list[int | float], rounds: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    low_idx = int(0.025 * (rounds - 1))
    high_idx = int(0.975 * (rounds - 1))
    return {
        "mean": statistics.mean(estimates),
        "std": statistics.pstdev(estimates) if len(estimates) > 1 else 0.0,
        "ci95_low": estimates[low_idx],
        "ci95_high": estimates[high_idx],
    }


def summarize_predictions(rows: list[dict[str, Any]], architecture: str, bootstrap: int, seed: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("pred_normalized") is not None and not row.get("error")]
    errors = [row for row in rows if row.get("error")]
    y_true = [row["gold_normalized"] for row in valid]
    y_pred = [row["pred_normalized"] for row in valid]
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: (x not in GRADE_LABELS, x))
    if set(labels).issubset(set(GRADE_LABELS)):
        labels = GRADE_LABELS
    elif set(labels).issubset(set(YES_NO_LABELS)):
        labels = YES_NO_LABELS

    grade_rows = [row for row in valid if row["gold_normalized"] in GRADE_LABELS]
    grade_true = [row["gold_normalized"] for row in grade_rows]
    grade_pred = [row["pred_normalized"] for row in grade_rows]

    correct_values = [1 if row["correct"] else 0 for row in valid]
    latency_values = [row["latency_s"] for row in valid if row.get("latency_s") is not None]

    summary = {
        "architecture": architecture,
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "error_examples": [row.get("error", "") for row in errors[:5]],
        "error_counts": dict(Counter(row.get("error", "") for row in errors)),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_bootstrap": bootstrap_metric(correct_values, bootstrap, seed),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "quadratic_weighted_kappa_grade_only": quadratic_weighted_kappa(grade_true, grade_pred, GRADE_LABELS),
        "mean_latency_s": statistics.mean(latency_values) if latency_values else 0.0,
        "median_latency_s": statistics.median(latency_values) if latency_values else 0.0,
        "label_distribution_gold": dict(Counter(y_true)),
        "label_distribution_pred": dict(Counter(y_pred)),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "auroc_note": (
            "AUROC is not computed because the current model APIs return hard labels, not class probabilities. "
            "Add probability/confidence scores to compute AUROC/RDR AUROC."
        ),
    }
    return summary


def summarize_rag(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rag_rows = [row for row in rows if row.get("architecture") == "rag" and not row.get("error")]
    if not rag_rows:
        return {}

    def mean_field(name: str) -> float:
        vals = [row[name] for row in rag_rows if row.get(name) is not None]
        return statistics.mean(vals) if vals else 0.0

    return {
        "retrieval_recall_at_k": mean_field("retrieval_recall_at_k"),
        "retrieval_precision_at_k": mean_field("retrieval_precision_at_k"),
        "mrr": mean_field("mrr"),
        "ndcg_at_k": mean_field("ndcg_at_k"),
        "mean_top_similarity": mean_field("top_similarity"),
        "retrieval_coverage": safe_div(sum(row.get("retrieved_count", 0) > 0 for row in rag_rows), len(rag_rows)),
        "answer_agreement_with_retrieved_majority": mean_field("retrieved_majority_agreement"),
        "rag_error_rate": safe_div(sum(1 for row in rows if row.get("architecture") == "rag" and row.get("error")), len(rows)),
        "safety_flag_rate": safe_div(sum(row.get("safety_flags_count", 0) > 0 for row in rag_rows), len(rag_rows)),
        "confidence_distribution": dict(Counter(row.get("rag_confidence_level", "") for row in rag_rows)),
    }


def relevance_metrics(
    retrieved: list[dict[str, Any]],
    gold: str,
    filter_type: str,
    k: int,
    project_question_type: str,
) -> dict[str, Any]:
    relevances = []
    answers = []
    for rec in retrieved[:k]:
        rec_answers = []
        for pair in rec.get("qa_pairs", []):
            if pair.get("question_type") == filter_type:
                pair_question = str(pair.get("question", "")).lower()
                if project_question_type == "inside_optic" and "optic" not in pair_question:
                    continue
                if project_question_type == "inside_exudates" and "optic" in pair_question:
                    continue
                rec_answers.append(normalize_answer(pair.get("answer")))
        if not rec_answers and filter_type == "inside":
            rec_answers = [normalize_answer(pair.get("answer")) for pair in rec.get("qa_pairs", []) if pair.get("question_type") == "inside"]
        relevant = int(gold in rec_answers)
        relevances.append(relevant)
        answers.extend(rec_answers)

    recall = 1.0 if any(relevances) else 0.0
    precision = safe_div(sum(relevances), len(relevances))
    reciprocal_rank = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            reciprocal_rank = 1.0 / idx
            break
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
    ideal = sum(1 / math.log2(idx + 2) for idx in range(min(sum(relevances), k)))
    ndcg = safe_div(dcg, ideal)
    majority = Counter(answers).most_common(1)[0][0] if answers else None
    return {
        "retrieval_recall_at_k": recall,
        "retrieval_precision_at_k": precision,
        "mrr": reciprocal_rank,
        "ndcg_at_k": ndcg,
        "retrieved_majority_answer": majority,
        "retrieved_majority_agreement": 1.0 if majority == gold else 0.0,
    }


def run_vqa(row: dict[str, Any], image: Image.Image, modules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    result = modules["model_inference"].run_inference(image, row["project_question_type"])
    return normalize_answer(result.get("raw_answer")), {"raw_model_answer": result.get("raw_answer")}


def run_rag(
    row: dict[str, Any],
    image: Image.Image,
    modules: dict[str, Any],
    user_id: int,
    top_k: int,
    alpha: float,
) -> tuple[str, dict[str, Any]]:
    filter_map = {
        "whole": "whole",
        "fovea": "fovea",
        "grade": "grade",
        "inside_exudates": "inside",
        "inside_optic": "inside",
    }
    filter_type = filter_map.get(row["project_question_type"], row["project_question_type"])

    question_types = modules["model_inference"].QUESTION_TYPES
    question = question_types[row["project_question_type"]]["question"]

    # Do not call rag_pipeline() during batch evaluation. The application pipeline
    # contains user-facing guardrails such as rate limiting, which makes long
    # evaluation runs fail. This direct path measures the actual multimodal RAG
    # architecture: CLIP image+text retrieval followed by the VQA answer model.
    retrieved = modules["rag_engine"].search(
        query_image=image,
        query_text=question,
        top_k=top_k,
        alpha=alpha,
        filter_type=filter_type,
    )
    result = modules["model_inference"].run_inference(image, row["project_question_type"])
    predicted = normalize_answer(result.get("raw_answer"))

    retrieval = relevance_metrics(
        retrieved,
        row["gold_normalized"],
        filter_type,
        top_k,
        row["project_question_type"],
    )
    top_similarity = retrieved[0].get("score") if retrieved else None
    retrieved_scores = [rec.get("score", 0.0) for rec in retrieved]
    mean_similarity = statistics.mean(retrieved_scores) if retrieved_scores else 0.0
    extra = {
        "retrieved_count": len(retrieved),
        "top_similarity": top_similarity,
        "mean_similarity": mean_similarity,
        "raw_model_answer": result.get("raw_answer"),
        "rag_confidence_level": retrieval_confidence_label(top_similarity, len(retrieved)),
        "safety_flags_count": 0,
        **retrieval,
    }
    return predicted, extra


def retrieval_confidence_label(top_similarity: float | None, retrieved_count: int) -> str:
    if top_similarity is None or retrieved_count == 0:
        return "No Retrieval"
    if top_similarity >= 0.85 and retrieved_count >= 3:
        return "High"
    if top_similarity >= 0.70 and retrieved_count >= 2:
        return "Moderate"
    return "Low"


def evaluate_architecture(
    architecture: str,
    dataset: list[dict[str, Any]],
    image_dir: str,
    modules: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(dataset, start=1):
        eval_id = f"{row.get('image_name')}::{row.get('question')}"
        started = time.time()
        output = {
            "eval_id": eval_id,
            "architecture": architecture,
            "image_name": row.get("image_name"),
            "question_type": row.get("project_question_type"),
            "question": row.get("question"),
            "gold": row.get("answer"),
            "gold_normalized": row["gold_normalized"],
            "pred_normalized": None,
            "correct": False,
            "latency_s": None,
            "error": "",
        }
        try:
            image_path = resolve_image(image_dir, row["image_name"])
            image = Image.open(image_path).convert("RGB")
            if architecture == "vqa":
                pred, extra = run_vqa(row, image, modules)
            else:
                pred, extra = run_rag(row, image, modules, args.user_id, args.top_k, args.alpha)
            output.update(extra)
            output["pred_normalized"] = pred
            output["correct"] = pred == row["gold_normalized"]
        except Exception as exc:
            output["error"] = str(exc)
        finally:
            output["latency_s"] = round(time.time() - started, 4)
            rows.append(output)
        print(
            f"[{architecture}] {idx}/{len(dataset)} "
            f"{output['question_type']} gold={output['gold_normalized']} pred={output['pred_normalized']} "
            f"ok={output['correct']} err={bool(output['error'])}"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    add_project_to_path(args.project_dir)

    modules = {
        "model_inference": importlib.import_module("model_inference"),
        "rag_engine": importlib.import_module("rag_engine"),
    }
    configure_project_modules(modules, args.project_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.qa_file, args.sample_size, args.seed, args.skip_inside)
    if not dataset:
        raise RuntimeError("No usable evaluation rows found.")

    architectures = ["vqa", "rag"] if args.architecture == "both" else [args.architecture]
    all_predictions = []
    summary = {
        "config": vars(args),
        "dataset_size": len(dataset),
        "architectures": {},
        "comparison": {},
        "multimodal_rag_metrics_note": (
            "Retrieval relevance is estimated by whether retrieved same-question QA pairs have the same answer. "
            "For stronger RAG evaluation, add expert relevance judgments and free-text reference explanations."
        ),
    }

    by_arch = {}
    for architecture in architectures:
        rows = evaluate_architecture(architecture, dataset, args.image_dir, modules, args)
        by_arch[architecture] = rows
        all_predictions.extend(rows)
        arch_summary = summarize_predictions(rows, architecture, args.bootstrap, args.seed)
        if architecture == "rag":
            arch_summary["multimodal_rag"] = summarize_rag(rows)
        summary["architectures"][architecture] = arch_summary
        write_matrix(
            output_dir / f"confusion_matrix_{architecture}.csv",
            arch_summary["labels"],
            arch_summary["confusion_matrix"],
        )
        write_csv(output_dir / f"per_class_{architecture}.csv", arch_summary["per_class"])

    if "vqa" in by_arch and "rag" in by_arch:
        summary["comparison"]["mcnemar_p_value_accuracy_vqa_vs_rag"] = mcnemar_p_value(by_arch["vqa"], by_arch["rag"])

    write_csv(output_dir / "predictions.csv", all_predictions)
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved evaluation outputs to:", output_dir.resolve())
    print(json.dumps(summary["architectures"], indent=2))


if __name__ == "__main__":
    main()
