"""FastAPI service: POST an image to /predict, get JSON text back."""
import io
import os
import time
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL = os.getenv("MODEL_NAME", "microsoft/trocr-base-handwritten")
ADAPTER = os.getenv("ADAPTER_PATH")  # optional LoRA adapter folder
MAX_BYTES = 5 * 1024 * 1024

app = FastAPI(title="MarathiScribe AI")
processor = TrOCRProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL)
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "adapter": ADAPTER}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Only PNG or JPEG images are accepted.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 5 MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image.")
    t0 = time.perf_counter()
    pixel = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel, max_length=64)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
