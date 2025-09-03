# ML/server.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np  # NumPy 타입 확인용
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

warnings.filterwarnings(action="ignore")

# -----------------------------------------------------------------------------
# 실행 컨텍스트 대응:
# - 권장: python -m ML.server (패키지 실행)
# - 보조: python ML/server.py (직접 실행 시에도 임시로 패키지 루트 추가)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ML_DIR = THIS_FILE.parent                 # .../ML
PROJECT_ROOT = ML_DIR.parent              # 프로젝트 루트

if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

# 패키지 절대/상대 import (모듈 실행 권장)
try:
    # 패키지로 실행될 때
    from .KoBERTModel.ensemble_utils import ensemble_inference
    from .deepvoice_detection.predict_deepvoice import deepvoice_predict
    from .speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
except Exception:
    # 직접 실행 시 임시 fallback (가능하면 모듈 실행 사용!)
    from ML.KoBERTModel.ensemble_utils import ensemble_inference
    from ML.deepvoice_detection.predict_deepvoice import deepvoice_predict
    from ML.speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio

# -----------------------------------------------------------------------------
# Flask APP
# -----------------------------------------------------------------------------
# 템플릿을 ML/templates 하위에 둔다고 가정 (다른 위치면 수정)
TEMPLATE_DIR = (ML_DIR / "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR.as_posix())  # main.html 렌더링
CORS(app)

log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.logger.info("Flask App Logger initialized.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device} in server.py")

# -----------------------------------------------------------------------------
# 경로/모델 파일
# -----------------------------------------------------------------------------
DEEPVOICE_MODEL_FILENAME = "best_f1_model.pt"
DEEPVOICE_MODEL_PATH = ML_DIR / "deepvoice_detection" / "model" / DEEPVOICE_MODEL_FILENAME
DEEPVOICE_CONFIG_PATH = ML_DIR / "deepvoice_detection" / "deepvoice_config.json"

critical_error = False
if not DEEPVOICE_MODEL_PATH.exists():
    app.logger.critical(f"Deepvoice model NOT FOUND at {DEEPVOICE_MODEL_PATH}")
    critical_error = True
if not DEEPVOICE_CONFIG_PATH.exists():
    app.logger.critical(f"Deepvoice config NOT FOUND at {DEEPVOICE_CONFIG_PATH}")
    critical_error = True
if critical_error:
    app.logger.error("Essential model or config files are missing (deepvoice). "
                     "Audio-related features may be disabled.")

UPLOAD_DIR = ML_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def safe_float(x: Any, default: float = 0.0) -> float:
    """숫자/문자/NumPy 타입 모두 안전하게 float로 변환."""
    try:
        # NumPy 타입 방어
        if isinstance(x, (np.generic,)):
            return float(np.asarray(x))
        return float(x)
    except Exception:
        return default


def build_text_response(analysis_result: Dict[str, Any], text_fallback: str) -> Dict[str, Any]:
    """/predict 응답 JSON을 UI가 기대하는 키들로 구성."""
    llm_score = safe_float(analysis_result.get("llm_score", 0))
    final_label = analysis_result.get("final_label", "알 수 없음 (분석 실패)")
    text_out = analysis_result.get("text", text_fallback)

    return {
        "final_label": final_label,
        "text": text_out,
        "llm_score": round(llm_score, 2),
        "voice_score": 0,
        "deepfake_score": "N/A",
        "total_score": round(llm_score, 2),  # 텍스트만 있을 때 total = llm
    }

# -----------------------------------------------------------------------------
# 라우트
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def main_page():
    # ML/templates/main.html 존재 가정
    if TEMPLATE_DIR.exists() and (TEMPLATE_DIR / "main.html").exists():
        return render_template("main.html")
    # 템플릿이 없으면 간단 메시지
    return "Server is running. (No main.html found)", 200


@app.route("/predict", methods=["POST"])
def predict_text_route():
    """
    텍스트만 입력 받아 KoBERT 기반 LLM 추론.
    UI가 기대하는 키(final_label, text, llm_score, voice_score, deepfake_score, total_score) 보장.
    """
    try:
        data = request.get_json(silent=True) or {}
        text_input = (data.get("text") or "").strip()

        if not text_input:
            return jsonify({
                "final_label": "입력 없음",
                "text": "(입력된 텍스트가 없습니다)",
                "llm_score": 0,
                "voice_score": 0,
                "deepfake_score": "N/A",
                "total_score": 0,
                "error": "텍스트 입력이 필요합니다."
            }), 400

        analysis_result = ensemble_inference(text_input) or {}
        response_data = build_text_response(analysis_result, text_input)
        app.logger.debug(f"Text prediction API response data: {response_data}")
        return jsonify(response_data), 200

    except Exception:
        app.logger.error("텍스트 예측 API 오류", exc_info=True)
        return jsonify({
            "final_label": "오류 발생",
            "text": "(오류)",
            "llm_score": 0,
            "voice_score": 0,
            "deepfake_score": "N/A",
            "total_score": 0,
            "error": "텍스트 분석 중 서버 오류가 발생했습니다."
        }), 500


@app.route("/api/audio_result", methods=["POST"])
def api_audio_result():
    """
    오디오 업로드 → (다중/단일 화자) 분석.
    - 다중 화자: analyze_multi_speaker_audio()가 반환한 dict를 UI 키 체계에 맞춰 보정/확장
    - 단일 화자: STT → 텍스트 분석 + 딥보이스 확률 → UI 키 구성
    """
    app.logger.debug("Received request for /api/audio_result")

    # 파일 유무/확장자 검사
    if "audio_file" not in request.files:
        app.logger.warning("Audio file not in request.files")
        return jsonify({"error": "오디오 파일이 필요합니다."}), 400

    audio_file = request.files["audio_file"]
    if not audio_file.filename or not audio_file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        app.logger.warning(f"Invalid audio file: {audio_file.filename}")
        return jsonify({"error": "유효한 오디오 파일(.wav, .mp3, .flac)을 업로드하세요."}), 400

    # 저장
    filename = secure_filename(audio_file.filename)
    audio_path = (UPLOAD_DIR / filename).resolve()
    audio_file.save(audio_path.as_posix())
    app.logger.debug(f"Audio file saved to {audio_path}")

    # 다중/단일 화자 옵션
    multi_speaker = True
    try:
        multi_speaker_form_value = request.form.get("multi_speaker", "true")
        multi_speaker = multi_speaker_form_value.lower() == "true"
    except Exception:
        pass
    app.logger.debug(f"Multi-speaker mode: {multi_speaker}")

    # 기본 응답 골격(오류 대비)
    final_response_data: Dict[str, Any] = {
        "speaker_0": {
            "deepfake_score": 0.0,
            "final_label": "분석 실패",
            "llm_score": 0.0,
            "text": "(분석 중 오류 발생 또는 데이터 없음)",
            "total_score": 0.0,
            "voice_score": 0.0,
            "phishing": False,
            "final_decision": "분석 실패",
            "phishing_detected_text": False,
            "deepfake_detected_voice": False,
        }
    }

    try:
        # Deepvoice 파일 없으면 음성 관련 기능 비활성
        deepvoice_ok = DEEPVOICE_MODEL_PATH.exists() and DEEPVOICE_CONFIG_PATH.exists()
        if not deepvoice_ok:
            app.logger.warning("Deepvoice model/config is missing. Voice-related scores will be zeros.")

        if multi_speaker:
            app.logger.debug(f"Analyzing multi-speaker audio: {audio_path}")
            # 파이프라인 결과 예시:
            # {
            #   "speaker_me":   {"text": ..., "text_score": 0~100, "phishing_detected_text": bool,
            #                    "deepfake_score": 0~1, "deepfake_detected_voice": bool, ...},
            #   "speaker_other": {...}
            # }
            raw_results = analyze_multi_speaker_audio(
                audio_path.as_posix(),
                DEEPVOICE_MODEL_PATH.as_posix(),
                DEEPVOICE_CONFIG_PATH.as_posix()
            ) if deepvoice_ok else analyze_multi_speaker_audio(
                audio_path.as_posix(), None, None
            )
            app.logger.debug(f"Multi-speaker analysis raw result: {raw_results}")

            if isinstance(raw_results, dict) and raw_results:
                processed: Dict[str, Any] = {}
                for speaker_id, data in raw_results.items():
                    llm_s  = safe_float(data.get("text_score", 0.0))     # 0~100
                    dv_prob = safe_float(data.get("deepfake_score", 0.0)) # 0~1

                    # 0~100 스케일 + 8:2 융합
                    voice_s = round(dv_prob * 100, 2)
                    total_s = round((0.8 * llm_s) + (0.2 * voice_s), 2)

                    # 개별 플래그(있으면 사용, 없으면 합리적 기본)
                    text_flag  = bool(data.get("phishing_detected_text", llm_s >= 70))
                    voice_flag = bool(data.get("deepfake_detected_voice", dv_prob >= 0.5))

                    # 최종 이진 판정: 70점 이상이면 피싱
                    final_label = "피싱" if total_s >= 70 else "정상"

                    processed[speaker_id] = {
                        "text": data.get("text", "(STT 결과 없음)"),
                        "phishing_detected_text": text_flag,
                        "text_score": round(llm_s, 2),
                        "deepfake_score": round(dv_prob, 4),  # 원본 확률(0~1)
                        "deepfake_detected_voice": voice_flag,
                        "phishing": (total_s >= 70),
                        "final_decision": final_label,

                        # UI 호환 키
                        "final_label": final_label,
                        "llm_score": round(llm_s, 2),
                        "voice_score": voice_s,     # 0~100
                        "total_score": total_s,     # 0~100
                    }
                final_response_data = processed
            else:
                app.logger.warning("No valid multi-speaker results. Returning default error payload.")
                final_response_data["speaker_0"]["error"] = "화자 분석 결과를 처리할 수 없습니다."
                final_response_data = {"speaker_0": final_response_data["speaker_0"]}

        else:
            app.logger.debug(f"Analyzing single-speaker audio: {audio_path}")
            # --- STT ---
            try:
                # 지연 import (의존성/초기화 비용 절약)
                from .speaker_analysis.whisper_stt import transcribe_segment
            except Exception:
                from ML.speaker_analysis.whisper_stt import transcribe_segment  # fallback

            try:
                text = transcribe_segment(audio_path.as_posix())
            except Exception as e_stt:
                app.logger.error(f"STT Error for single speaker: {e_stt}", exc_info=True)
                text = "(STT 실패)"

            # --- 텍스트 LLM ---
            kobert_text_result = ensemble_inference(text) or {}
            llm_s = safe_float(kobert_text_result.get("llm_score", 0.0))
            is_text_phishing = bool(kobert_text_result.get("phishing_detected", llm_s > 50))

            # --- 딥보이스 ---
            if deepvoice_ok:
                deep_prob = deepvoice_predict(
                    audio_path.as_posix(),
                    DEEPVOICE_MODEL_PATH.as_posix(),
                    DEEPVOICE_CONFIG_PATH.as_posix()
                )
                deep_prob = 0.0 if deep_prob is None else safe_float(deep_prob, 0.0)
            else:
                deep_prob = 0.0

            is_voice_deepfake = deep_prob > 0.5

            # 0~100 스케일 + 8:2 융합
            voice_s = round(deep_prob * 100, 2)                 # 0~100
            total_s = round((0.8 * llm_s) + (0.2 * voice_s), 2) # 0~100

            # 최종 라벨 (70점 임계)
            final_label = "피싱" if total_s >= 70 else "정상"

            final_response_data = {
                "speaker_0": {
                    "deepfake_score": round(deep_prob, 4),
                    "final_label": final_label,
                    "llm_score": round(llm_s, 2),
                    "text": text,
                    "total_score": total_s,
                    "voice_score": voice_s,
                    "phishing_detected_text": is_text_phishing,
                    "deepfake_detected_voice": is_voice_deepfake,
                    "phishing": (total_s >= 70),
                    "final_decision": final_label,
                }
            }

        app.logger.debug(f"Final response data for audio API: {final_response_data}")
        return jsonify(final_response_data), 200

    except Exception:
        app.logger.error("오디오 API 처리 중 심각한 오류 발생", exc_info=True)
        error_response_key = "error_info"
        final_response_data = {
            error_response_key: {
                "error": "오디오 처리 중 서버 오류가 발생했습니다.",
                "final_label": "처리 오류",
                "text": "(오류)",
                "llm_score": 0,
                "voice_score": 0,
                "deepfake_score": 0.0,
                "total_score": 0,
            }
        }
        return jsonify(final_response_data), 500

    finally:
        # 업로드 파일 정리
        try:
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)
        except Exception as e_remove:
            app.logger.error(f"Error cleaning up audio file {audio_path}: {e_remove}")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 개발 시
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    # 배포 시 (예)
    # app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)