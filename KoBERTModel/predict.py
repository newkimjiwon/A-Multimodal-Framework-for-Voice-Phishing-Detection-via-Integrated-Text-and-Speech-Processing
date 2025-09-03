# 파일명: predict.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# KoBERT 분류기
class KoBERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_classes=2, dr_rate=0.4):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        output_dropped = self.dropout(pooled_output)
        logits = self.classifier(output_dropped)
        return logits

# 데이터셋 클래스
class KoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 예측 함수
def predict(text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 현재 파일 기준 절대 경로 설정
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "KoBERTModel" / "model" / "train.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    logger.info(f"모델 로드: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    bert_model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)

    model = KoBERTClassifier(bert_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = KoBERTDataset([text], [0], tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        batch = next(iter(loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)
        phishing_prob = probs[0][1].item()

    score = round(phishing_prob * 100, 2)
    is_phishing = score > 50

    return {
        "text": text,
        "llm_score": score,
        "phishing_detected": is_phishing
    }

if __name__ == "__main__":
    sample_text = "안녕하세요, 계좌번호를 알려주세요."
    result = predict(sample_text)
    print(result)
