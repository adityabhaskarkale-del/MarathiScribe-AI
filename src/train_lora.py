"""LoRA fine-tuning of TrOCR (frozen vision encoder, LoRA on decoder attention)."""
import argparse
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="microsoft/trocr-base-handwritten")
p.add_argument("--train", default="data/train.csv")
p.add_argument("--val", default="data/val.csv")
p.add_argument("--out", default="adapter")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--lr", type=float, default=5e-4)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

for prm in model.encoder.parameters():  # frozen vision tower
    prm.requires_grad = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"]))
model.print_trainable_parameters()
model.to(device)


class OCRData(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        pixel = processor(images=img, return_tensors="pt").pixel_values[0]
        ids = processor.tokenizer(str(row["text"]), padding="max_length",
                                  max_length=64, truncation=True).input_ids
        ids = [t if t != processor.tokenizer.pad_token_id else -100 for t in ids]
        return {"pixel_values": pixel, "labels": torch.tensor(ids)}


train_dl = DataLoader(OCRData(args.train), batch_size=args.batch, shuffle=True)
val_dl = DataLoader(OCRData(args.val), batch_size=args.batch)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    total = 0.0
    for step, b in enumerate(train_dl):
        loss = model(pixel_values=b["pixel_values"].to(device),
                     labels=b["labels"].to(device)).loss
        (loss / args.accum).backward()
        total += loss.item()
        if (step + 1) % args.accum == 0:
            opt.step()
            opt.zero_grad()
    model.eval()
    vtotal = 0.0
    with torch.no_grad():
        for b in val_dl:
            vtotal += model(pixel_values=b["pixel_values"].to(device),
                            labels=b["labels"].to(device)).loss.item()
    print(f"epoch {epoch + 1}: train_loss={total / len(train_dl):.4f} "
          f"val_loss={vtotal / len(val_dl):.4f}")  # val rising while train falls = overfitting

model.save_pretrained(args.out)
print("Saved LoRA adapter to", args.out)
