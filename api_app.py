"""
FastAPI backend for DeepEye.

Run:
    uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload

Default first admin, created only when the users table is empty:
    username: admin
    password: admin123

Override with DEEPEYE_ADMIN_USERNAME and DEEPEYE_ADMIN_PASSWORD.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import uuid
from difflib import SequenceMatcher
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from database import (
    delete_user_by_username,
    delete_query,
    delete_rag_search,
    get_user_queries,
    get_user_rag_searches,
    init_db,
    list_users,
    login_user,
    logout_user,
    register_user,
    save_query,
    save_rag_search,
    validate_token_details,
)
from guardrails import sanitize_query, validate_image, validate_query
from model_inference import GRADE_LABELS, QUESTION_TYPES, run_inference


APP_NAME = "DeepEye API"
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
EXPLAIN_PATTERNS = [
    "explain",
    "explain image",
    "explain the image",
    "describe",
    "describe image",
    "analyze",
    "analyse",
    "analysis",
    "full report",
    "all questions",
]

QUESTION_ALIASES = {
    "whole": [
        "hard exudates in this image",
        "exudates in image",
        "exudates present",
        "yellow lesions",
        "bright lesions",
    ],
    "fovea": [
        "hard exudates in the fovea",
        "fovea exudates",
        "macula exudates",
        "central retina exudates",
    ],
    "grade": [
        "dme grade",
        "diabetic macular edema grade",
        "severity",
        "stage",
        "classify dme",
    ],
    "inside_exudates": [
        "hard exudates in this region",
        "exudates in region",
        "inside region exudates",
        "mask exudates",
    ],
    "inside_optic": [
        "optic disc in this region",
        "optic discs in this region",
        "optic disk",
        "disc in region",
        "mask optic",
    ],
}

PATHOLOGY_EXPLANATIONS = {
    "whole": "Hard exudates are bright lipid deposits that can appear in diabetic retinal disease.",
    "fovea": "Foveal hard exudates are clinically important because the fovea supports central vision.",
    "grade": "DME grade summarizes the likely severity of diabetic macular edema from the image.",
    "inside_exudates": "The region-focused answer checks whether the highlighted area contains hard exudates.",
    "inside_optic": "The region-focused answer checks whether the highlighted area corresponds to the optic disc.",
}


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Professional FastAPI backend for DeepEye DME VQA and multimodal RAG.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DEEPEYE_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

security = HTTPBearer()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: dict[str, Any]


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=6, max_length=200)
    is_admin: bool = False


class ApiMessage(BaseModel):
    message: str


@app.on_event("startup")
def startup() -> None:
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> dict[str, Any]:
    user = validate_token_details(credentials.credentials)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    return user


def current_admin(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
    if not user["is_admin"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return user


def _read_upload(upload: UploadFile) -> tuple[Image.Image, bytes]:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type. Upload a fundus JPG/PNG/BMP/TIFF image.")
    raw = upload.file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read image. Upload a valid image file.")
    return image, raw


def _save_upload(raw: bytes, filename: str, user_id: int) -> str:
    ext = os.path.splitext(filename or "")[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        ext = ".jpg"
    path = os.path.join(UPLOAD_DIR, f"u{user_id}_{uuid.uuid4().hex[:10]}{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _looks_like_fundus(image: Image.Image) -> bool:
    arr = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = arr.mean(axis=2)
    warm_pixels = (r > g * 0.9) & (g > b * 0.9) & (brightness > 25)
    warm_ratio = float(warm_pixels.mean())
    non_dark_ratio = float((brightness > 30).mean())
    channel_spread = float(np.mean(np.abs(r - b)))
    return warm_ratio > 0.18 and non_dark_ratio > 0.20 and channel_spread > 10


def _is_explain_request(message: str) -> bool:
    msg = message.lower().strip()
    return any(pattern in msg for pattern in EXPLAIN_PATTERNS)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_question_type(message: str) -> str | None:
    msg = sanitize_query(message).lower()
    scores: dict[str, float] = {}
    for key, info in QUESTION_TYPES.items():
        candidates = [info["question"], *QUESTION_ALIASES[key]]
        lexical = max(_similarity(msg, candidate) for candidate in candidates)
        keyword_hits = sum(1 for alias in QUESTION_ALIASES[key] if any(tok in msg for tok in alias.split()))
        scores[key] = lexical + min(keyword_hits * 0.03, 0.18)

    best_key = max(scores, key=scores.get)
    if scores[best_key] < 0.42:
        return None
    return best_key


def _answer_label(result: dict[str, Any]) -> str:
    if result["question_type"] == "grade":
        return GRADE_LABELS.get(result["raw_answer"], result["answer_label"])
    return "Yes" if result["raw_answer"] == "yes" else "No"


def _pathology_note(result: dict[str, Any]) -> str:
    base = PATHOLOGY_EXPLANATIONS[result["question_type"]]
    if result["question_type"] == "grade":
        grade = result["raw_answer"]
        severity = GRADE_LABELS.get(grade, f"Grade {grade}")
        return f"{base} Model output: {severity}."
    presence = "present" if result["raw_answer"] == "yes" else "not clearly present"
    return f"{base} In this model run, the feature is {presence}."


def _format_vqa_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": result["question_type"],
        "matched_standard_question": result["question"],
        "answer": {
            "raw": result["raw_answer"],
            "label": _answer_label(result),
        },
        "pathological_feature_explanation": _pathology_note(result),
        "mask": {
            "required": bool(result["needs_mask"]),
            "applied": bool(result["mask_applied"]),
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


@app.get("/vqa/questions")
def supported_questions(_: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
    return {
        key: {
            "question": value["question"],
            "description": value["description"],
            "needs_mask": value["needs_mask"],
            "answers": value["answers"],
        }
        for key, value in QUESTION_TYPES.items()
    }


@app.get("/rag/status")
def rag_status(_: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
    from rag_engine import index_status

    return index_status()


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    ok, token_or_message, user_id = login_user(payload.username, payload.password)
    if not ok:
        raise HTTPException(status_code=401, detail=token_or_message)
    user = validate_token_details(token_or_message) or {"id": user_id, "username": payload.username, "is_admin": False}
    return LoginResponse(access_token=token_or_message, user=user)


@app.post("/auth/logout")
def logout(user: Annotated[dict[str, Any], Depends(current_user)], credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]) -> ApiMessage:
    logout_user(credentials.credentials)
    return ApiMessage(message="Logged out successfully.")


@app.get("/auth/me")
def me(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
    return user


@app.get("/admin/users")
def admin_users(_: Annotated[dict[str, Any], Depends(current_admin)]) -> list[dict[str, Any]]:
    return list_users()


@app.post("/admin/users", status_code=201)
def admin_create_user(
    payload: CreateUserRequest,
    _: Annotated[dict[str, Any], Depends(current_admin)],
) -> dict[str, Any]:
    ok, msg = register_user(payload.username, payload.password, payload.is_admin)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"message": msg, "username": payload.username, "is_admin": payload.is_admin}


@app.delete("/admin/users/{username}")
def admin_delete_user(
    username: str,
    admin: Annotated[dict[str, Any], Depends(current_admin)],
) -> ApiMessage:
    ok, msg = delete_user_by_username(username, requester_id=admin["id"])
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return ApiMessage(message=msg)


def _chat_pipeline(
    *,
    user: dict[str, Any],
    image: UploadFile,
    message: str,
    mask: UploadFile | None,
    architecture: Literal["vqa", "multimodal_rag"],
    top_k: int = 5,
    alpha: float = 0.7,
) -> dict[str, Any]:
    clean_message = sanitize_query(message)
    qv = validate_query(clean_message)
    if not qv.passed:
        raise HTTPException(status_code=400, detail=qv.message)

    img, raw = _read_upload(image)
    iv = validate_image(img, len(raw))
    if not iv.passed:
        raise HTTPException(status_code=400, detail=iv.message)
    if not _looks_like_fundus(img):
        raise HTTPException(status_code=400, detail="Image does not appear to be a retinal fundus image.")

    mask_img = None
    if mask is not None:
        mask_img, _ = _read_upload(mask)
        mask_img = mask_img.convert("L")

    explain_all = _is_explain_request(clean_message)
    if explain_all:
        question_types = list(QUESTION_TYPES.keys())
        mode = "explain_all"
    else:
        matched = match_question_type(clean_message)
        if not matched:
            raise HTTPException(
                status_code=400,
                detail="Question is not related to the five supported DME VQA questions.",
            )
        question_types = [matched]
        mode = "single_question"

    saved_path = _save_upload(raw, image.filename, user["id"])
    answers = []
    for question_type in question_types:
        result = run_inference(img, question_type, mask=mask_img)
        answers.append(_format_vqa_result(result))
        save_query(user["id"], saved_path, result["question"], result["answer_label"])

    rag = None
    if architecture == "multimodal_rag":
        from rag_engine import rag_pipeline

        rag_outputs = []
        for question_type in question_types:
            out, retrieved, err = rag_pipeline(
                user_id=user["id"],
                image=img,
                question_type=question_type,
                file_size_b=len(raw),
                top_k=max(1, min(top_k, 10)),
                alpha=max(0.0, min(alpha, 1.0)),
                mask=mask_img,
            )
            if err:
                rag_outputs.append({"question_type": question_type, "error": err})
                continue
            compact_retrieved = [
                {
                    "image_name": r.get("image_name"),
                    "grade": r.get("grade"),
                    "has_exudates": r.get("has_exudates"),
                    "fovea": r.get("fovea"),
                    "score": r.get("score"),
                }
                for r in retrieved
            ]
            save_rag_search(
                user["id"],
                saved_path,
                QUESTION_TYPES[question_type]["question"],
                json.dumps(compact_retrieved),
                json.dumps(dataclasses.asdict(out)),
            )
            rag_outputs.append(
                {
                    "question_type": question_type,
                    "structured_answer": dataclasses.asdict(out),
                    "retrieved_cases": compact_retrieved,
                }
            )
        rag = rag_outputs

    return {
        "status": "success",
        "architecture": architecture,
        "architecture_label": "Multimodal RAG" if architecture == "multimodal_rag" else "VQA",
        "mode": mode,
        "input_question": clean_message,
        "image_path": saved_path,
        "answers": answers,
        "rag": rag,
        "disclaimer": "AI-generated research output only. Consult an ophthalmologist for clinical decisions.",
    }


@app.post("/vqa/chat")
def vqa_chat(
    user: Annotated[dict[str, Any], Depends(current_user)],
    image: Annotated[UploadFile, File()],
    message: Annotated[str, Form(..., min_length=1, max_length=500)],
    mask: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    return _chat_pipeline(
        user=user,
        image=image,
        message=message,
        mask=mask,
        architecture="vqa",
    )


@app.post("/rag/chat")
def multimodal_rag_chat(
    user: Annotated[dict[str, Any], Depends(current_user)],
    image: Annotated[UploadFile, File()],
    message: Annotated[str, Form(..., min_length=1, max_length=500)],
    mask: Annotated[UploadFile | None, File()] = None,
    top_k: Annotated[int, Form()] = 5,
    alpha: Annotated[float, Form()] = 0.7,
) -> dict[str, Any]:
    return _chat_pipeline(
        user=user,
        image=image,
        message=message,
        mask=mask,
        architecture="multimodal_rag",
        top_k=top_k,
        alpha=alpha,
    )


@app.get("/history/vqa")
def vqa_history(user: Annotated[dict[str, Any], Depends(current_user)]) -> list[dict[str, Any]]:
    return get_user_queries(user["id"])


@app.delete("/history/vqa/{query_id}")
def delete_vqa_history(
    query_id: int,
    user: Annotated[dict[str, Any], Depends(current_user)],
) -> ApiMessage:
    if not delete_query(query_id, user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ApiMessage(message="Conversation deleted.")


@app.get("/history/rag")
def rag_history(user: Annotated[dict[str, Any], Depends(current_user)]) -> list[dict[str, Any]]:
    return get_user_rag_searches(user["id"])


@app.delete("/history/rag/{search_id}")
def delete_rag_history(
    search_id: int,
    user: Annotated[dict[str, Any], Depends(current_user)],
) -> ApiMessage:
    if not delete_rag_search(search_id, user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ApiMessage(message="Conversation deleted.")
