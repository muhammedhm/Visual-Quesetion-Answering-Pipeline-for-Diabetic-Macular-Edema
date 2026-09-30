"""
Evaluate ROUGE-L and BERTScore for DeepEyeNet VQA and Multimodal RAG.

This is a standalone evaluation script. It does not modify the application.

Quick examples from deepeye_2_way:
    ..\\venv\\Scripts\\python.exe evaluate_text_metrics_rouge_bertscore.py --architecture both --sample-size 100
    ..\\venv\\Scripts\\python.exe evaluate_text_metrics_rouge_bertscore.py --architecture rag --sample-size 500

If BERTScore is missing:
    ..\\venv\\Scripts\\python.exe -m pip install bert-score

Outputs:
    text_metrics_summary.json
    text_metric_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


GRADE_TEXT = {
    "0": "Grade 0. No diabetic macular edema is indicated.",
    "1": "Grade 1. Mild diabetic macular edema is indicated.",
    "2": "Grade 2. Moderate to severe diabetic macular edema is indicated.",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir.parent / "dme_vqa"
    parser = argparse.ArgumentParser(description="Evaluate ROUGE-L and BERTScore for VQA and Multimodal RAG.")
    parser.add_argument("--project-dir", default=str(script_dir))
    parser.add_argument("--qa-file", default=str(dataset_root / "qa" / "trainqa.json"))
    parser.add_argument("--image-dir", default=str(dataset_root / "visual" / "train"))
    parser.add_argument("--output-dir", default=str(script_dir.parent / "deepeye_text_metric_outputs"))
    parser.add_argument("--architecture", choices=["vqa", "rag", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=0, help="0 evaluates all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--bert-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-inside", action="store_true")
    args = parser.parse_args()

    missing = []
    for label, path in [("project-dir", args.project_dir), ("qa-file", args.qa_file), ("image-dir", args.image_dir)]:
        if not Path(path).exists():
            missing.append(f"--{label} path not found: {path}")
    if missing:
        parser.error("\n".join(missing))
    return args


def add_project_to_path(project_dir: str) -> None:
    project = str(Path(project_dir).resolve())
    if project not in sys.path:
        sys.path.insert(0, project)


def configure_project_paths(modules: dict[str, Any], project_dir: str) -> None:
    project = Path(project_dir).resolve()
    modules["rag_engine"].INDEX_PATH = str(project / "rag_index" / "faiss.index")
    modules["rag_engine"].META_PATH = str(project / "rag_index" / "metadata.pkl")


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    for grade in ["0", "1", "2"]:
        if text == grade or f"grade {grade}" in text:
            return grade
    return text


def question_type(row: dict[str, Any]) -> str | None:
    qt = str(row.get("question_type", "")).lower()
    question = str(row.get("question", "")).lower()
    if qt in {"whole", "fovea", "grade"}:
        return qt
    if qt == "inside":
        return "inside_optic" if "optic" in question or "disc" in question else "inside_exudates"
    if qt in {"inside_optic", "inside_exudates"}:
        return qt
    return None


def load_dataset(path: str, sample_size: int, seed: int, skip_inside: bool) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    usable = []
    for row in rows:
        qt = question_type(row)
        if not qt:
            continue
        if skip_inside and qt.startswith("inside"):
            continue
        row = dict(row)
        row["project_question_type"] = qt
        row["gold_normalized"] = normalize_answer(row.get("answer"))
        usable.append(row)
    if sample_size and sample_size < len(usable):
        rng = random.Random(seed)
        rng.shuffle(usable)
        usable = usable[:sample_size]
    return usable


def reference_text(row: dict[str, Any]) -> str:
    gold = row["gold_normalized"]
    qt = row["project_question_type"]
    if qt == "grade":
        return GRADE_TEXT.get(gold, f"Grade {gold}.")
    if qt == "whole":
        feature = "Hard exudates are present in the retinal fundus image." if gold == "yes" else "Hard exudates are not present in the retinal fundus image."
    elif qt == "fovea":
        feature = "Hard exudates are present in the foveal region." if gold == "yes" else "Hard exudates are not present in the foveal region."
    elif qt == "inside_optic":
        feature = "The optic disc is present in the highlighted region." if gold == "yes" else "The optic disc is not present in the highlighted region."
    else:
        feature = "Hard exudates are present in the highlighted region." if gold == "yes" else "Hard exudates are not present in the highlighted region."
    return feature


def prediction_text_from_answer(question_type_value: str, pred: str) -> str:
    if question_type_value == "grade":
        return GRADE_TEXT.get(pred, f"Grade {pred}.")
    if question_type_value == "whole":
        return "Hard exudates are present in the retinal fundus image." if pred == "yes" else "Hard exudates are not present in the retinal fundus image."
    if question_type_value == "fovea":
        return "Hard exudates are present in the foveal region." if pred == "yes" else "Hard exudates are not present in the foveal region."
    if question_type_value == "inside_optic":
        return "The optic disc is present in the highlighted region." if pred == "yes" else "The optic disc is not present in the highlighted region."
    return "Hard exudates are present in the highlighted region." if pred == "yes" else "Hard exudates are not present in the highlighted region."


def resolve_image(image_dir: str, image_name: str) -> Path:
    direct = Path(image_dir) / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = Path(image_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {image_name}")


def lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_f1": f1}


def rag_filter_type(project_question_type: str) -> str:
    return {
        "whole": "whole",
        "fovea": "fovea",
        "grade": "grade",
        "inside_exudates": "inside",
        "inside_optic": "inside",
    }.get(project_question_type, project_question_type)


def run_vqa(row: dict[str, Any], image: Image.Image, modules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    result = modules["model_inference"].run_inference(image, row["project_question_type"])
    pred = normalize_answer(result.get("raw_answer"))
    return prediction_text_from_answer(row["project_question_type"], pred), {"pred_normalized": pred}


def run_rag(row: dict[str, Any], image: Image.Image, modules: dict[str, Any], top_k: int, alpha: float) -> tuple[str, dict[str, Any]]:
    question_info = modules["model_inference"].QUESTION_TYPES[row["project_question_type"]]
    retrieved = modules["rag_engine"].search(
        query_image=image,
        query_text=question_info["question"],
        top_k=top_k,
        alpha=alpha,
        filter_type=rag_filter_type(row["project_question_type"]),
    )
    result = modules["model_inference"].run_inference(image, row["project_question_type"])
    pred = normalize_answer(result.get("raw_answer"))
    retrieved_count = len(retrieved)
    top_score = retrieved[0].get("score") if retrieved else None
    text = prediction_text_from_answer(row["project_question_type"], pred)
    if retrieved:
        text += f" Retrieved {retrieved_count} visually similar retinal cases for context."
    return text, {"pred_normalized": pred, "retrieved_count": retrieved_count, "top_similarity": top_score}


def evaluate_architecture(
    architecture: str,
    dataset: list[dict[str, Any]],
    image_dir: str,
    modules: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(dataset, start=1):
        started = time.time()
        output = {
            "architecture": architecture,
            "image_name": row.get("image_name"),
            "question_type": row["project_question_type"],
            "question": row.get("question"),
            "reference_text": reference_text(row),
            "prediction_text": "",
            "gold_normalized": row["gold_normalized"],
            "pred_normalized": "",
            "exact_answer_match": False,
            "latency_s": 0.0,
            "error": "",
        }
        try:
            image = Image.open(resolve_image(image_dir, row["image_name"])).convert("RGB")
            if architecture == "vqa":
                pred_text, extra = run_vqa(row, image, modules)
            else:
                pred_text, extra = run_rag(row, image, modules, args.top_k, args.alpha)
            output["prediction_text"] = pred_text
            output.update(extra)
            output["exact_answer_match"] = output["pred_normalized"] == output["gold_normalized"]
            output.update(rouge_l(output["prediction_text"], output["reference_text"]))
        except Exception as exc:
            output["error"] = str(exc)
            output.update({"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0})
        output["latency_s"] = round(time.time() - started, 4)
        rows.append(output)
        print(f"[{architecture}] {idx}/{len(dataset)} rougeL={output['rouge_l_f1']:.3f} error={bool(output['error'])}")
    return rows


def add_bertscore(rows: list[dict[str, Any]], model_name: str, batch_size: int) -> None:
    valid = [row for row in rows if not row.get("error") and row.get("prediction_text") and row.get("reference_text")]
    if not valid:
        return
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore package is not installed. Install it with: "
            "python -m pip install bert-score"
        ) from exc

    predictions = [row["prediction_text"] for row in valid]
    references = [row["reference_text"] for row in valid]
    precision, recall, f1 = score(
        predictions,
        references,
        model_type=model_name,
        lang="en",
        batch_size=batch_size,
        verbose=True,
        rescale_with_baseline=False,
    )
    for row, p, r, f in zip(valid, precision.tolist(), recall.tolist(), f1.tolist()):
        row["bertscore_precision"] = p
        row["bertscore_recall"] = r
        row["bertscore_f1"] = f


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.mean(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], architecture: str) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    errors = [row for row in rows if row.get("error")]
    return {
        "architecture": architecture,
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_errors": len(errors),
        "error_examples": [row["error"] for row in errors[:5]],
        "error_counts": dict(Counter(row["error"] for row in errors)),
        "exact_answer_accuracy": mean(valid, "exact_answer_match"),
        "rouge_l_precision": mean(valid, "rouge_l_precision"),
        "rouge_l_recall": mean(valid, "rouge_l_recall"),
        "rouge_l_f1": mean(valid, "rouge_l_f1"),
        "bertscore_precision": mean(valid, "bertscore_precision"),
        "bertscore_recall": mean(valid, "bertscore_recall"),
        "bertscore_f1": mean(valid, "bertscore_f1"),
        "mean_latency_s": mean(valid, "latency_s"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    add_project_to_path(args.project_dir)
    modules = {
        "model_inference": importlib.import_module("model_inference"),
        "rag_engine": importlib.import_module("rag_engine"),
    }
    configure_project_paths(modules, args.project_dir)

    dataset = load_dataset(args.qa_file, args.sample_size, args.seed, args.skip_inside)
    architectures = ["vqa", "rag"] if args.architecture == "both" else [args.architecture]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {
        "config": vars(args),
        "metric_note": (
            "ROUGE-L and BERTScore are text-overlap/semantic-similarity metrics. "
            "For this dataset, references are generated from the ground-truth VQA labels because "
            "the QA file contains short labels rather than full human explanations."
        ),
        "architectures": {},
    }

    for architecture in architectures:
        rows = evaluate_architecture(architecture, dataset, args.image_dir, modules, args)
        if not args.skip_bertscore:
            add_bertscore(rows, args.bert_model, args.bert_batch_size)
        summary["architectures"][architecture] = summarize(rows, architecture)
        all_rows.extend(rows)

    write_csv(output_dir / "text_metric_predictions.csv", all_rows)
    with open(output_dir / "text_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved text metric outputs to:", output_dir.resolve())
    print(json.dumps(summary["architectures"], indent=2))


if __name__ == "__main__":
    main()
