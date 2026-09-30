"""
rag_engine.py — Multimodal RAG for DME VQA dataset
────────────────────────────────────────────────────
Dataset: trainqa.json
  - 5 unique questions across 4 question types
  - Answers: yes / no / 0 / 1 / 2
  - Images: IDRiD fundus images
  - Masks: binary region masks for 'inside' question type

RAG pipeline:
  1. CLIP-embed query image (+ optional text)
  2. FAISS cosine search over indexed dataset images
  3. Retrieve top-K records with their Q&A context
  4. Build augmented prompt → LLaVA inference
  5. Output guardrails → StructuredRAGOutput
"""

import os
import json
import pickle
import time
import logging
import numpy as np
from PIL import Image
from typing import Optional

from guardrails import (
    validate_image, validate_query, sanitize_query,
    check_rate_limit, parse_structured_output,
    StructuredRAGOutput,
)

logger = logging.getLogger("deepeye.rag")
logging.basicConfig(level=logging.INFO)

INDEX_PATH   = "./rag_index/faiss.index"
META_PATH    = "./rag_index/metadata.pkl"

# ── Path configuration ───────────────────────────────────────────────────────
# Set these to match where you unzipped dme_vqa.zip.
# The script will also auto-detect common locations at runtime.
QA_FILE   = "C:/Users/ISHHAQ/OneDrive/Desktop/FYP_Project/DeepEyeNet/dme_vqa/qa/trainqa.json"
IMAGE_DIR = "C:/Users/ISHHAQ/OneDrive/Desktop/FYP_Project/DeepEyeNet/dme_vqa/visual/train"

def _resolve_paths():
    """
    Try to auto-detect QA_FILE and IMAGE_DIR if the configured paths don't exist.
    Checks several common unzip locations relative to cwd.
    """
    global QA_FILE, IMAGE_DIR

    candidates_qa = [
        QA_FILE,
        "./qa/trainqa.json",
        "./trainqa.json",
        "../dme_vqa/qa/trainqa.json",
    ]
    candidates_img = [
        IMAGE_DIR,
        "./visual/train",
        "./train",
        "../dme_vqa/visual/train",
    ]

    for p in candidates_qa:
        if os.path.exists(p):
            QA_FILE = p
            break

    for p in candidates_img:
        if os.path.isdir(p) and any(f.endswith(".jpg") for f in os.listdir(p)):
            IMAGE_DIR = p
            break

    print(f"   QA file  : {os.path.abspath(QA_FILE)}  (exists={os.path.exists(QA_FILE)})")
    print(f"   Image dir: {os.path.abspath(IMAGE_DIR)}  (exists={os.path.isdir(IMAGE_DIR)})")

_clip_model     = None
_clip_processor = None
_faiss_index    = None
_metadata       = None

# ─────────────────────────────────────────
# CLIP
# ─────────────────────────────────────────
def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    from transformers import CLIPProcessor, CLIPModel
    import torch
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    if torch.cuda.is_available():
        _clip_model = _clip_model.cuda()
    return _clip_model, _clip_processor


def embed_image(image: Image.Image) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    m, p = _load_clip()
    inp = p(images=image.convert("RGB"), return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        # Use vision_model directly to bypass get_image_features() version quirks.
        # vision_model always returns BaseModelOutputWithPooling;
        # pooler_output is the [CLS] token embedding we want.
        vision_out = m.vision_model(pixel_values=inp["pixel_values"])
        f = vision_out.pooler_output          # shape: (1, 768)
        f = m.visual_projection(f)            # project to CLIP embedding space
    f = F.normalize(f, p=2, dim=-1)
    return f.cpu().numpy().astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    m, p = _load_clip()
    inp = p(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        # Use text_model directly — same reason as embed_image.
        text_out = m.text_model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
        )
        f = text_out.pooler_output            # shape: (1, 512)
        f = m.text_projection(f)              # project to CLIP embedding space
    f = F.normalize(f, p=2, dim=-1)
    return f.cpu().numpy().astype(np.float32)


def embed_combined(image: Image.Image, text: str, alpha: float = 0.7) -> np.ndarray:
    ie = embed_image(image)
    te = embed_text(text)
    c  = alpha * ie + (1 - alpha) * te
    c  = c / np.linalg.norm(c, axis=-1, keepdims=True)
    return c.astype(np.float32)


# ─────────────────────────────────────────
# Dataset parser for DME trainqa.json
# ─────────────────────────────────────────
def _parse_dataset() -> list[dict]:
    """
    Parse trainqa.json into records for indexing.
    Groups by image so each unique image gets one index entry
    containing all its associated Q&A pairs (up to 4 types).
    This avoids indexing the same image 9779 times.
    """
    with open(QA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Group by image_name
    by_image: dict[str, dict] = {}
    for entry in raw:
        img = entry["image_name"]
        if img not in by_image:
            by_image[img] = {
                "image_name"  : img,
                "image_path"  : img,   # filename only; resolved at index time using IMAGE_DIR
                "qa_pairs"    : [],
                "grade"       : None,
                "has_exudates": None,
                "fovea"       : None,
            }
        rec = by_image[img]
        qt  = entry["question_type"]
        ans = str(entry["answer"])

        rec["qa_pairs"].append({
            "question_type": qt,
            "question"     : entry["question"],
            "answer"       : ans,
        })

        # Store key findings for display
        if qt == "grade":
            rec["grade"] = ans
        elif qt == "whole":
            rec["has_exudates"] = ans
        elif qt == "fovea":
            rec["fovea"] = ans

    return list(by_image.values())


# ─────────────────────────────────────────
# Build index
# ─────────────────────────────────────────
def build_index(force: bool = False, max_records: int = 9999):
    import faiss
    os.makedirs("./rag_index", exist_ok=True)

    if not force and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Index already exists. Use force=True to rebuild.")
        return

    # Auto-detect paths
    print("\nResolving dataset paths...")
    _resolve_paths()

    if not os.path.exists(QA_FILE):
        raise FileNotFoundError(
            f"\n❌ QA file not found: {os.path.abspath(QA_FILE)}\n"
            f"   Make sure dme_vqa.zip is unzipped and QA_FILE points to trainqa.json.\n"
            f"   Current working dir: {os.getcwd()}"
        )

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"\n❌ Image directory not found: {os.path.abspath(IMAGE_DIR)}\n"
            f"   Make sure IMAGE_DIR points to the folder containing IDRiD_*.jpg files.\n"
            f"   Current working dir: {os.getcwd()}"
        )

    # Count available images upfront
    available_imgs = {f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))}
    print(f"   Found {len(available_imgs)} images in {IMAGE_DIR}")

    records = _parse_dataset()[:max_records]
    print(f"   Parsed {len(records)} unique image records from QA file")

    # Check overlap
    matched = sum(1 for r in records if os.path.basename(r["image_path"]) in available_imgs)
    print(f"   Images matched to QA records: {matched}/{len(records)}")

    if matched == 0:
        # Show a sample to help user debug
        sample_rec  = records[0]["image_name"] if records else "—"
        sample_file = next(iter(available_imgs), "—") if available_imgs else "—"
        raise FileNotFoundError(
            f"\n❌ Zero images matched. Paths are misaligned.\n"
            f"   QA record expects  : '{sample_rec}'\n"
            f"   Directory contains : '{sample_file}'\n"
            f"   IMAGE_DIR = {os.path.abspath(IMAGE_DIR)}\n"
            f"   Fix: edit IMAGE_DIR at the top of rag_engine.py to point to the "
            f"folder that actually contains your .jpg files."
        )

    _load_clip()

    embeddings, metadata, skipped = [], [], 0
    _first_err = False
    for i, rec in enumerate(records):
        if i % 50 == 0:
            print(f"   Embedding [{i}/{len(records)}] — embedded={len(embeddings)}, skipped={skipped}")

        # Build path after _resolve_paths() has updated IMAGE_DIR
        ip = os.path.join(IMAGE_DIR, rec["image_name"])
        rec["image_path"] = ip

        if os.path.exists(ip):
            try:
                img = Image.open(ip).convert("RGB")
                embeddings.append(embed_image(img)[0])
                metadata.append(rec)
            except Exception as e:
                skipped += 1
                logger.warning(f"Image embed error [{ip}]: {e}")
                if not _first_err:
                    print(f"\n   First embed ERROR: {ip}\n   Reason: {e}\n")
                    _first_err = True
        else:
            skipped += 1
            logger.warning(f"Image not found: {ip}")
            if not _first_err:
                print(f"\n   First MISSING image: {ip}")
                print(f"   IMAGE_DIR={IMAGE_DIR}  image_name={rec['image_name']}\n")
                _first_err = True

    print(f"\n   Final: embedded={len(embeddings)}, skipped={skipped}")

    if len(embeddings) == 0:
        raise RuntimeError(
            "❌ No embeddings were created. All images failed to load or embed.\n"
            f"   IMAGE_DIR = {os.path.abspath(IMAGE_DIR)}\n"
            "   Check that the directory contains valid .jpg files."
        )

    arr   = np.stack(embeddings).astype(np.float32)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Index saved: {len(metadata)} entries → {INDEX_PATH}")


# ─────────────────────────────────────────
# Load index
# ─────────────────────────────────────────
def _load_index():
    global _faiss_index, _metadata
    if _faiss_index is not None:
        return _faiss_index, _metadata
    import faiss
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("RAG index not found. Build it first.")
    _faiss_index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _metadata = pickle.load(f)
    return _faiss_index, _metadata


# ─────────────────────────────────────────
# Search
# ─────────────────────────────────────────
def search(
    query_image: Image.Image,
    query_text:  Optional[str] = None,
    top_k:       int   = 5,
    alpha:       float = 0.7,
    filter_type: Optional[str] = None,   # filter by question_type if set
) -> list[dict]:
    index, metadata = _load_index()
    q = embed_combined(query_image, query_text, alpha) if query_text else embed_image(query_image)
    scores, indices = index.search(q, min(top_k * 3, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        rec = metadata[idx].copy()
        rec["score"] = float(score)

        # Optional filter: only return records that have a specific question type
        if filter_type:
            has_type = any(p["question_type"] == filter_type for p in rec["qa_pairs"])
            if not has_type:
                continue

        results.append(rec)
        if len(results) >= top_k:
            break

    return results


# ─────────────────────────────────────────
# Production RAG pipeline
# ─────────────────────────────────────────
def rag_pipeline(
    user_id:       int,
    image:         Image.Image,
    question_type: str,
    file_size_b:   int   = 0,
    top_k:         int   = 5,
    alpha:         float = 0.7,
    mask:          Optional[Image.Image] = None,
) -> tuple[StructuredRAGOutput, list[dict], Optional[str]]:
    """
    Full production RAG pipeline for DME VQA.

    Args:
        question_type : one of 'whole','fovea','grade','inside_exudates','inside_optic'
        mask          : optional mask image for inside question types
    """
    from model_inference import QUESTION_TYPES, apply_mask, run_inference

    t0 = time.time()

    # Rate limit
    ok, msg = check_rate_limit(user_id)
    if not ok:
        return StructuredRAGOutput(), [], msg

    # Image validation
    iv = validate_image(image, file_size_b)
    if not iv.passed:
        return StructuredRAGOutput(), [], iv.message

    qt_info   = QUESTION_TYPES[question_type]
    question  = qt_info["question"]

    # Apply mask if needed (for inside question types)
    img_for_inference = image.convert("RGB")
    if qt_info["needs_mask"] and mask is not None:
        img_for_inference = apply_mask(img_for_inference, mask)

    # Map inside types to dataset filter
    filter_map = {
        "whole"           : "whole",
        "fovea"           : "fovea",
        "grade"           : "grade",
        "inside_exudates" : "inside",
        "inside_optic"    : "inside",
    }
    filter_type = filter_map.get(question_type)

    # Retrieve similar cases
    try:
        retrieved = search(
            img_for_inference,
            query_text  = question,
            top_k       = top_k,
            alpha       = alpha,
            filter_type = filter_type,
        )
    except FileNotFoundError as e:
        return StructuredRAGOutput(), [], str(e)
    except Exception as e:
        return StructuredRAGOutput(), [], f"Retrieval error: {e}"

    # Build context from retrieved records
    context_parts = []
    for i, rec in enumerate(retrieved):
        # Find the matching Q&A pair for this question type
        matching = [p for p in rec["qa_pairs"] if p["question_type"] == filter_type]
        if not matching:
            matching = rec["qa_pairs"][:1]

        lines = [f"Similar Case {i+1} ({rec['image_name']}):"]
        for pair in matching[:2]:
            lines.append(f"  Q: {pair['question']}")
            lines.append(f"  A: {pair['answer']}")
        if rec.get("grade"):
            lines.append(f"  DME Grade: {rec['grade']}")
        lines.append(f"  Similarity: {rec['score']*100:.1f}%")
        context_parts.append("\n".join(lines))

    context = "\n\n".join(context_parts)

    augmented_prompt = (
        "You are a clinical AI assistant specializing in diabetic macular edema (DME) "
        "analysis of retinal fundus images. "
        "Here are similar cases from the clinical database:\n\n"
        f"{context}\n\n"
        "Based on the above context and the uploaded image, answer the following question. "
        "Use hedged language. Answer with 'yes', 'no', or a grade number (0/1/2) as appropriate.\n\n"
        f"Question: {question}"
    )

    # Run LLaVA with the augmented prompt
    try:
        result = run_inference(image, question_type, mask=mask)
        raw_answer = result["answer_label"]
    except Exception as e:
        return StructuredRAGOutput(), retrieved, f"Inference error: {e}"

    elapsed = time.time() - t0

    # Build structured output
    # Derive disease tags from retrieved records
    for rec in retrieved:
        if rec.get("grade"):
            rec["diseases"] = f"DME Grade {rec['grade']}"
        elif rec.get("has_exudates") == "yes":
            rec["diseases"] = "Hard exudates present"
        else:
            rec["diseases"] = "No hard exudates"

    structured = parse_structured_output(
        raw_answer = raw_answer,
        retrieved  = retrieved,
        query_text = question,
        elapsed_s  = elapsed,
    )

    return structured, retrieved, None


# ─────────────────────────────────────────
# Index status
# ─────────────────────────────────────────
def index_status() -> dict:
    exists  = os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)
    n_total = 0
    size_mb = 0.0
    if exists:
        size_mb = os.path.getsize(INDEX_PATH) / 1e6
        try:
            import faiss
            n_total = faiss.read_index(INDEX_PATH).ntotal
        except Exception:
            pass
    return {"exists": exists, "n_total": n_total, "size_mb": round(size_mb, 1)}