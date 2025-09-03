# shared_model_loader.py

from transformers import BertModel

# 공통 KoBERT 모델 인스턴스 (모든 모듈에서 공유)
shared_bert = BertModel.from_pretrained("monologg/kobert")
shared_bert.eval()  # 추론 전용