# 파일명: ML/KoBERTModel/ensemble_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 경로 고정: 이 파일 기준으로 모델 가중치 위치를 잡아 실행 위치(CWD) 영향 제거
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent               # .../ML/KoBERTModel
MODEL_PATH = THIS_DIR / "model" / "train.pt"             # .../ML/KoBERTModel/model/train.pt
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# (train.py와 호환) 분류기 정의
# ---------------------------------------------------------------------
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
        # 일부 모델은 pooler_output이 없을 수 있으므로 방어
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            # [CLS] 토큰 임베딩 사용
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            pooled_output = last_hidden[:, 0, :]     # [B, H]
        attentions = outputs.attentions

        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits, attentions

class KoBERTDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int = 128):
        self.texts = [str(t) for t in texts]
        self.labels = labels if labels is not None else [0] * len(self.texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        # kobert 토크나이저는 token_type_ids가 있을 수도/없을 수도 있음 → 방어
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            result["token_type_ids"] = torch.zeros_like(result["input_ids"])
        return result

# ---------------------------------------------------------------------
# 전역 로딩(토크나이저/베이스모델/학습된 분류기)
# ---------------------------------------------------------------------
kobert_tokenizer_global: Optional[AutoTokenizer] = None
kobert_trained_classifier_global: Optional[KoBERTClassifier] = None
kobert_model_load_error_global: Optional[str] = None

try:
    logger.info("[ensemble_utils] Loading KoBERT tokenizer...")
    kobert_tokenizer_global = AutoTokenizer.from_pretrained(
        "monologg/kobert", trust_remote_code=True
    )

    logger.info("[ensemble_utils] Loading KoBERT base (output_attentions=True)...")
    bert_base_for_classifier = AutoModel.from_pretrained(
        "monologg/kobert", trust_remote_code=True, output_attentions=True
    )

    if MODEL_PATH.exists():
        logger.info(f"[ensemble_utils] Loading trained classifier: {MODEL_PATH}")
        model = KoBERTClassifier(bert_base_for_classifier).to(device)
        state = torch.load(MODEL_PATH.as_posix(), map_location=device)
        model.load_state_dict(state)
        model.eval()
        kobert_trained_classifier_global = model
        logger.info("✅ [ensemble_utils] KoBERT trained classifier loaded successfully.")
    else:
        kobert_model_load_error_global = f"KoBERT trained classifier file not found at: {MODEL_PATH}"
        logger.critical(kobert_model_load_error_global)

except Exception as e_load:
    kobert_model_load_error_global = f"Error loading KoBERT model: {e_load}"
    logger.critical(kobert_model_load_error_global, exc_info=True)

# ---------------------------------------------------------------------
# 내부 예측 함수
# ---------------------------------------------------------------------
@torch.no_grad()
def kobert_predict_internal(text: str) -> Tuple[float, Any]:
    """
    Returns:
        phishing_probability (float in [0,1]),
        attentions (Any or None)
    """
    if kobert_model_load_error_global or kobert_tokenizer_global is None or kobert_trained_classifier_global is None:
        logger.error(f"KoBERT model not loaded. Error: {kobert_model_load_error_global}")
        return 0.0, None

    dataset = KoBERTDataset([text], [0], kobert_tokenizer_global, max_len=128)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    logits, attentions = kobert_trained_classifier_global(input_ids, attention_mask, token_type_ids)
    probs = F.softmax(logits, dim=-1)
    phishing_prob = float(probs[0, 1].detach().cpu().item())  # class 1 = phishing
    return phishing_prob, attentions

# ---------------------------------------------------------------------
# 서버가 기대하는 키로 맞춘 공개 함수
# ---------------------------------------------------------------------
def _label_from_score(score_0_100: float) -> str:
    """UI 규칙에 맞춘 간단 라벨링."""
    if score_0_100 >= 90:
        return "확정"
    if score_0_100 >= 70:
        return "위험"
    if score_0_100 > 50:
        return "의심"
    return "아님"

def ensemble_inference(text: str) -> Dict[str, Any]:
    """
    server.py가 기대하는 딕셔너리 형식으로 반환:
    {
        "text": str,
        "llm_score": float(0~100),
        "final_label": str,                 # "확정"/"위험"/"의심"/"아님"
        "phishing_detected": bool,          # (llm_score > 50 기준)
        "attentions": Any (옵션)
        # 필요시 추가 필드 가능
    }
    """
    phishing_prob, attentions = kobert_predict_internal(text)
    llm_score = round(phishing_prob * 100.0, 2)
    final_label = _label_from_score(llm_score)
    phishing_detected = llm_score > 50.0

    # 모델 미로딩 등의 근본 오류를 클라이언트에서도 파악할 수 있게 error 필드 추가(선택)
    payload: Dict[str, Any] = {
        "text": text,
        "llm_score": llm_score,
        "final_label": final_label,
        "phishing_detected": phishing_detected,
        "attentions": attentions,
    }
    if kobert_model_load_error_global:
        payload["error"] = kobert_model_load_error_global
    return payload
