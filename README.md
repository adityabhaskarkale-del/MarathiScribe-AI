# MarathiScribe AI - Marathi Handwriting Recognition

Fine-tunes a pretrained TrOCR model with **LoRA (Hugging Face PEFT)** to read Marathi (Devanagari)
handwriting, evaluates it with **CER / WER**, and serves it with **FastAPI + Docker**.

## Results
| Model | CER | WER | p50 latency | p95 latency |
|---|---|---|---|---|
| Baseline | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| LoRA fine-tuned | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

(Numbers come from `results.json` after running the steps below. Reference point: IndicOCR reports ~29.8% WER on Marathi handwriting.)

## Project layout
- `src/make_splits.py` - fixed 80/10/10 train/val/test split (seed 42)
- `src/train_lora.py` - LoRA training (rank 16, frozen vision encoder, gradient accumulation)
- `src/evaluate.py` - CER/WER + p50/p95 latency + error CSV
- `src/metrics.py` - metrics with Unicode NFC normalization
- `app/main.py` - FastAPI service (`/predict`, `/health`)
- `Dockerfile`

## How to run
1. `pip install -r requirements.txt`
2. Put images in `data/images/` and make `data/labels.csv` with columns `image_path,text`.
3. `python src/make_splits.py`
4. Baseline: `python src/evaluate.py --out baseline.json`
5. Train: `python src/train_lora.py --epochs 5`
6. Fine-tuned: `python src/evaluate.py --adapter adapter --out finetuned.json`
7. API: `uvicorn app.main:app --port 8000` then open http://localhost:8000/docs
8. Docker: `docker build -t marathiscribe .` and `docker run -p 8000:8000 marathiscribe`

Use `--model <hf-name>` / `MODEL_NAME` to switch to a Devanagari-capable TrOCR checkpoint.
