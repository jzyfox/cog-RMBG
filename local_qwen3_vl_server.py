from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MODEL_NAME = "Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_DIR = r"J:\Models\Qwen\Qwen3-VL-8B-Instruct"
MODEL_DIR_ENV = "QWEN3_VL_MODEL_DIR"
MODEL_DEVICE = "cuda:0"
MODEL_DTYPE_NAME = "bfloat16"
MAX_NEW_TOKENS = 768

REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

app = FastAPI(title="Local Qwen3-VL Service")

_MODEL: Any = None
_PROCESSOR: Any = None
_MODEL_DIR: Path | None = None
_GENERATION_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    image_data_url: str
    prompt: str
    model: str | None = None


class GenerateResponse(BaseModel):
    text: str
    model: str


def _resolve_model_dir() -> Path:
    model_dir = Path(os.getenv(MODEL_DIR_ENV, DEFAULT_MODEL_DIR)).expanduser().resolve()
    if not model_dir.exists():
        raise RuntimeError(f"Qwen3-VL model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise RuntimeError(f"Qwen3-VL model path is not a directory: {model_dir}")

    missing_files = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]
    if missing_files:
        missing_display = ", ".join(missing_files)
        raise RuntimeError(f"Qwen3-VL model directory is missing required files: {missing_display}")
    return model_dir


def _decode_data_url_to_image(image_data_url: str) -> Image.Image:
    if not isinstance(image_data_url, str) or "," not in image_data_url:
        raise ValueError("image_data_url must be a valid data URL")

    header, encoded = image_data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("image_data_url must use base64 encoding")

    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("image_data_url is not valid base64 data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("image_data_url does not contain a valid image") from exc


def _load_model() -> None:
    global _MODEL, _PROCESSOR, _MODEL_DIR

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local Qwen3-VL inference, but no CUDA device is available")

    _MODEL_DIR = _resolve_model_dir()
    _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_DIR)
    _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": MODEL_DEVICE},
    )
    _MODEL.eval()


def _ensure_loaded() -> tuple[Any, Any]:
    if _MODEL is None or _PROCESSOR is None or _MODEL_DIR is None:
        raise RuntimeError("Qwen3-VL model is not loaded")
    return _MODEL, _PROCESSOR


def _generate_text(image_data_url: str, prompt: str) -> str:
    import torch

    model, processor = _ensure_loaded()
    image = _decode_data_url_to_image(image_data_url)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(MODEL_DEVICE)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


@app.on_event("startup")
def startup_load_model() -> None:
    _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = _MODEL is not None and _PROCESSOR is not None and _MODEL_DIR is not None
    return {
        "loaded": loaded,
        "model": MODEL_NAME,
        "model_dir": str(_MODEL_DIR) if _MODEL_DIR is not None else None,
        "device": MODEL_DEVICE,
        "dtype": MODEL_DTYPE_NAME,
    }


def _handle_generate(payload: GenerateRequest) -> GenerateResponse:
    if not payload.image_data_url.strip():
        raise HTTPException(status_code=400, detail="image_data_url is required")
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        with _GENERATION_LOCK:
            text = _generate_text(payload.image_data_url, payload.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(text=text, model=MODEL_NAME)


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    return _handle_generate(payload)


@app.post("/semantic-tag", response_model=GenerateResponse)
def semantic_tag(payload: GenerateRequest) -> GenerateResponse:
    return _handle_generate(payload)
