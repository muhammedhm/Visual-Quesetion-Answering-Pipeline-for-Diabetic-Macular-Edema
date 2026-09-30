"""
model_inference.py
─────────────────
VQA inference for DME dataset model.

The model was trained on 5 specific questions:
  1. "Are there hard exudates in this image?"          (whole)
  2. "Are there hard exudates in the fovea?"           (fovea)
  3. "What is the diabetic macular edema grade for this image?" (grade)
  4. "are there hard exudates in this region?"         (inside)
  5. "are there optic discs in this region?"           (inside)

For 'inside' questions the mask is composited onto the image (same as
training): pixels OUTSIDE the mask region are dimmed to 40% brightness
so the model attends to the highlighted area.

Expected answers: "yes" | "no" | "0" | "1" | "2"
"""

import torch
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_LOCAL_DIR = "./model_cache/llava-qwen-0.5b"
MODEL_ID        = "llava-hf/llava-interleave-qwen-0.5b-hf"
ADAPTER_DIR     = "./llava-deepeye-adapter"

_model     = None
_processor = None

# ─────────────────────────────────────────
# Question catalogue — exactly as in training
# ─────────────────────────────────────────
QUESTION_TYPES = {
    "whole": {
        "question"    : "Are there hard exudates in this image?",
        "description" : "Detects hard exudates across the entire fundus image.",
        "needs_mask"  : False,
        "answers"     : ["yes", "no"],
    },
    "fovea": {
        "question"    : "Are there hard exudates in the fovea?",
        "description" : "Checks for hard exudates specifically in the foveal region.",
        "needs_mask"  : False,
        "answers"     : ["yes", "no"],
    },
    "grade": {
        "question"    : "What is the diabetic macular edema grade for this image?",
        "description" : "Grades DME severity: 0 = None, 1 = Mild, 2 = Moderate–Severe.",
        "needs_mask"  : False,
        "answers"     : ["0", "1", "2"],
    },
    "inside_exudates": {
        "question"    : "are there hard exudates in this region?",
        "description" : "Checks for hard exudates in a specific highlighted region (requires mask).",
        "needs_mask"  : True,
        "answers"     : ["yes", "no"],
    },
    "inside_optic": {
        "question"    : "are there optic discs in this region?",
        "description" : "Checks whether the optic disc falls in a specific region (requires mask).",
        "needs_mask"  : True,
        "answers"     : ["yes", "no"],
    },
}

# Human-readable answer expansions
GRADE_LABELS = {
    "0": "Grade 0 — No diabetic macular edema",
    "1": "Grade 1 — Mild diabetic macular edema",
    "2": "Grade 2 — Moderate to severe diabetic macular edema",
}


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Composite a binary mask onto the image.
    Pixels OUTSIDE the mask are dimmed to 40% brightness (same as training).
    """
    mask_l   = mask.convert("L").resize(image.size, Image.NEAREST)
    mask_arr = np.array(mask_l, dtype=np.float32) / 255.0
    binary   = (mask_arr > 0).astype(np.float32)
    img_arr  = np.array(image.convert("RGB"), dtype=np.float32)
    result   = np.where(binary[:, :, np.newaxis] == 1, img_arr, img_arr * 0.4)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    mp = MODEL_LOCAL_DIR if os.path.exists(MODEL_LOCAL_DIR) else MODEL_ID
    print("Loading VQA model...")

    _processor = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    if _processor.tokenizer.pad_token is None:
        _processor.tokenizer.pad_token = _processor.tokenizer.eos_token

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = LlavaForConditionalGeneration.from_pretrained(
        mp, quantization_config=qcfg, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, ADAPTER_DIR) if os.path.exists(ADAPTER_DIR) else base
    _model.eval()
    print("✅ VQA model ready")
    return _model, _processor


def run_inference(
    image:         Image.Image,
    question_type: str,
    mask:          Image.Image | None = None,
) -> dict:
    """
    Run DME VQA inference.

    Args:
        image         : uploaded fundus image
        question_type : one of QUESTION_TYPES keys
        mask          : optional PIL mask image for 'inside' question types

    Returns dict with:
        raw_answer    : model output token(s)
        answer_label  : human-readable answer
        question      : exact question text used
        question_type : the type key
        needs_mask    : whether this type required a mask
        mask_applied  : whether a mask was actually applied
    """
    model, processor = load_model()

    qt_info   = QUESTION_TYPES[question_type]
    question  = qt_info["question"]
    needs_mask = qt_info["needs_mask"]

    # Apply mask compositing if needed (exactly as in training)
    mask_applied = False
    img = image.convert("RGB")
    if needs_mask and mask is not None:
        img          = apply_mask(img, mask)
        mask_applied = True

    prompt = (
        f"<|im_start|>user\n"
        f"<image>\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = processor(text=prompt, images=img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,       # answers are very short: yes/no/0/1/2
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    n          = inputs["input_ids"].shape[1]
    raw_answer = processor.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip().lower()

    # Normalise to expected answer set
    expected   = qt_info["answers"]
    if raw_answer not in expected:
        # Find closest expected answer by substring match
        for exp in expected:
            if exp in raw_answer:
                raw_answer = exp
                break
        else:
            raw_answer = expected[0]   # fallback to first valid answer

    # Build human-readable label
    if question_type == "grade":
        answer_label = GRADE_LABELS.get(raw_answer, f"Grade {raw_answer}")
    else:
        answer_label = "✅ Yes" if raw_answer == "yes" else "❌ No"

    return {
        "raw_answer"   : raw_answer,
        "answer_label" : answer_label,
        "question"     : question,
        "question_type": question_type,
        "needs_mask"   : needs_mask,
        "mask_applied" : mask_applied,
        "processed_img": img,   # the image actually fed to model (with mask if applied)
    }
