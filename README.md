# Multimodal Voice Phishing Detection System

## 프로젝트 개요
본 프로젝트는 **텍스트와 음성 정보를 통합 분석하여 보이스피싱을 탐지**하는 멀티모달 프레임워크입니다. 텍스트 분류기(KoBERT 기반)와 합성음성 탐지기(CNN–BiLSTM 기반)를 결합하고, 화자 분리(Resemblyzer + K-Means), Whisper STT, 가중치 융합(텍스트:음성 = 8:2)을 통해 실제 통화 환경에서도 높은 신뢰도로 판별합니다.

## 주요 기능
- **텍스트 분석**: Whisper STT → KoBERT 분류
- **음성 분석**: MFCC → CNN–BiLSTM 합성음성 판별
- **화자 분리**: Resemblyzer 임베딩 + K-Means
- **멀티모달 융합**: 0.8 × Text + 0.2 × Voice 최종 점수 산출
- **시각화**: Self-Attention 기반 중요 토큰/패턴 시각화

## 프로젝트 구조 (간략)
```text
ML/
├── KoBERTModel/         # 텍스트 분류 모델 모듈
├── deepvoice_detection/ # 음성 합성 탐지 모듈
├── speaker_analysis/    # 화자 분리 및 STT 모듈
├── figure/              # 결과 시각화 자료
├── static/              # 데이터셋 및 로그(csv)
├── templates/           # 웹 인터페이스 템플릿
├── test/                # 실험 및 검증 스크립트
├── server.py            # 서버 실행 스크립트
├── shared_model_loader.py
├── requirements.txt
└── Dockerfile
```

## 성능 결과
- **Validation (200 문장)**: Accuracy, Precision, Recall, F1 모두 100%
- **실제 통화 (100건)**: 8:2 융합 모델이 가장 안정적
- **Attention 분석**: 정상 → 분산 패턴 / 피싱 → 금융·지시·긴급성 집중 패턴

## 아키텍처 (Screenshots & 설명)

### 전체 파이프라인 구조
<img width="4913" height="2268" alt="Architecture" src="https://github.com/user-attachments/assets/dea83e4a-d0a0-4b06-bc35-90d016b1db51" />
멀티모달 시스템 전체 구조를 나타냅니다. 입력된 음성은 화자 분리(Speaker Diarization) 단계를 거쳐 발화 단위로 분리되며, 각각 텍스트 분석(KoBERT)과 음성 분석(CNN–BiLSTM) 파이프라인을 병렬로 통과합니다. 이후 점수를 융합하여 최종 보이스피싱 여부를 판별합니다.

### 텍스트 분석 모델 (KoBERT 기반)
<img width="4913" height="2268" alt="LLM_Model" src="https://github.com/user-attachments/assets/d2cc1636-518d-43aa-a32e-df14416dd633" />
텍스트 입력은 KoBERT 임베딩을 통해 Self-Attention 기반 Transformer Encoder로 처리됩니다. 주어진 토큰 시퀀스는 어텐션 시각화 블록을 거쳐, Fully Connected Layer와 Softmax를 통해 피싱/정상 확률이 산출됩니다.

### 음성 분석 모델 (CNN–BiLSTM 기반)
<img width="4913" height="2268" alt="1D_CNN" src="https://github.com/user-attachments/assets/12fbb60e-3c33-479e-a3b8-1ca9e00c2adc" />
음성은 MFCC 특징으로 변환된 후, 1D CNN을 통해 지역적 패턴을 추출합니다. 이어서 BiLSTM Layer에서 시계열 의존성을 학습하며, Attention Layer를 통해 중요한 구간에 가중치를 부여합니다. 최종적으로 Fully Connected Layer에서 Sigmoid 함수를 사용해 합성/비합성 여부를 판정합니다.

## 한계점
- 화자 분리 성능이 잡음/음질에 영향을 받음
- 고정 가중치 융합 → 맥락 적응형 동적 융합 도입 필요

## 향후 과제
- Whisper 기반 정밀 다화자 분리 적용
- 동적 가중치 융합 기법 연구
- 다양한 피싱 시나리오 확장

## 참고
- 데이터: 금융감독원, AI Hub
- 관련 코드: [GitHub Repository](https://github.com/voicepaw/so-vits-svc-fork)
