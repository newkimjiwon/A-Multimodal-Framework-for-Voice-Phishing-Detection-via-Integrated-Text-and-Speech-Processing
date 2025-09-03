# ML/KoBERTModel/__init__.py

# 어떤 방식으로 실행하든 동작하도록 다중 시도
try:
    # 권장: 프로젝트 루트에서 `python -m ML.server` 로 실행 시
    from ML.shared_model_loader import shared_bert
except Exception:
    try:
        # 패키지 내부 상대 import (운영 환경에 따라 허용)
        from ..shared_model_loader import shared_bert
    except Exception:
        # ML 디렉터리에서 직접 실행하는 비표준 케이스
        from shared_model_loader import shared_bert
