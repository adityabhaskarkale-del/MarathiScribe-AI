"""Compute CER/WER and p50/p95 latency on the test split. Run baseline, then with --adapter."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from metrics import corpus_cer, corpus_wer

p = argparse.ArgumentParser()
p.add_argument("--model", default="microsoft/trocr-base-handwritten")
p.add_argument("--adapter", default=None)
p.add_argument("--test", default="data/test.csv")
p.add_argument("--out", default="results.json")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.to(device).eval()

df = pd.read_csv(args.test)
refs, hyps, times = [], [], []
for _, row in df.iterrows():
    img = Image.open(row["image_path"]).convert("RGB")
    pixel = processor(images=img, return_tensors="pt").pixel_values.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(pixel, max_length=64)
    times.append((time.perf_counter() - t0) * 1000)
    hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    refs.append(str(row["text"]))

res = {"model": args.model, "adapter": args.adapter, "n_images": len(df),
       "CER": round(corpus_cer(refs, hyps), 4), "WER": round(corpus_wer(refs, hyps), 4),
       "latency_ms_p50": round(float(np.percentile(times, 50)), 1),
       "latency_ms_p95": round(float(np.percentile(times, 95)), 1)}
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open(args.out, "w"), indent=2)
pd.DataFrame({"reference": refs, "prediction": hyps}).to_csv(
    args.out.replace(".json", "_errors.csv"), index=False)  # open this for error analysis
