# ML/KoBERTModel/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------
# 경로 고정(절대 경로): 이 파일 위치 기준으로 모두 계산
# -----------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent            # .../ML/KoBERTModel
ML_DIR   = THIS_DIR.parent                            # .../ML
PROJECT_ROOT = ML_DIR.parent                          # 프로젝트 루트
DATA_CSV = ML_DIR / "static" / "csv" / "KoBERT_dataset_v3.0.csv"
LOG_CSV  = ML_DIR / "static" / "csv" / "kobert_train_log.csv"
MODEL_PATH = THIS_DIR / "model" / "train.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# 로깅
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------------------------------------
# 시드 고정
# -----------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

set_seed(42)

# -----------------------------------------------------------
# 모델/데이터셋 정의
# -----------------------------------------------------------
class KoBERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size: int = 768, num_classes: int = 2, dr_rate: float = 0.4):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        # pooler_output이 없을 수 있으므로 방어
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits, outputs.attentions

class KoBERTDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 64):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------------------------------------
# 학습 루틴
# -----------------------------------------------------------
def train():
    logger.info(f"데이터셋 로딩: {DATA_CSV}")
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    df = df[df["Label"].astype(str).str.strip().isin(["0", "1"])]
    df["Label"] = df["Label"].astype(int)
    df = df[df["Transcript"].notnull()]
    df["Transcript"] = df["Transcript"].astype(str).str.strip()
    df = df[df["Transcript"] != ""]

    texts = df["Transcript"].tolist()
    labels = df["Label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    logger.info(f"훈련 데이터: {len(train_texts)}개, 검증 데이터: {len(val_texts)}개")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    bert = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True, output_attentions=True)
    model = KoBERTClassifier(bert).to(device)

    train_loader = DataLoader(KoBERTDataset(train_texts, train_labels, tokenizer), batch_size=32, shuffle=True)
    val_loader   = DataLoader(KoBERTDataset(val_texts,   val_labels,   tokenizer), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    logger.info("학습 시작...")
    training_logs: List[Dict[str, Any]] = []

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [훈련]", unit="batch")

        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels_t       = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).tolist())
            train_targets.extend(labels_t.tolist())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss = total_loss / max(1, len(train_loader))
        train_acc  = accuracy_score(train_targets, train_preds)
        train_f1   = f1_score(train_targets, train_preds)

        # 검증
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [검증]", unit="batch"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels_t       = batch["label"].to(device)
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                val_preds.extend(torch.argmax(logits, dim=1).tolist())
                val_targets.extend(labels_t.tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1  = f1_score(val_targets, val_preds)
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        training_logs.append({
            "Epoch": epoch + 1,
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc":  f"{train_acc:.4f}",
            "Train F1":   f"{train_f1:.4f}",
            "Validation Acc": f"{val_acc:.4f}",
            "Validation F1":  f"{val_f1:.4f}",
        })

    # 로그 저장
    pd.DataFrame(training_logs).to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"학습 로그 저장 완료: {LOG_CSV}")

    # 모델 저장(절대 경로)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"최종 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    train()
