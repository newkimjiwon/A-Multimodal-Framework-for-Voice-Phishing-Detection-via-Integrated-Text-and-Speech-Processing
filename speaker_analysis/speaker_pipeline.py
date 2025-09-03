# \ML\speaker_analysis\speaker_pipeline.py

from .diarization_utils import split_speakers
from .whisper_stt import transcribe_segment
from ..deepvoice_detection.predict_deepvoice import deepvoice_predict
from ..KoBERTModel.ensemble_utils import ensemble_inference

import os
import logging

logger = logging.getLogger(__name__)

# 이 함수가 server.py에서 임포트하여 호출하는 주된 파이프라인 함수입니다.
def analyze_multi_speaker_audio(audio_path: str, dv_model_path: str, dv_config_path: str):
    """
    오디오 파일을 입력받아 화자 분리, STT, 텍스트 분석, 딥보이스 탐지를 순차적으로 수행하고
    화자별 최종 분석 결과를 반환하는 메인 파이프라인 함수입니다.
    """
    logger.info(f"Starting audio analysis pipeline for: {audio_path}")
    logger.debug(f"Using DV Model: {dv_model_path}, DV Config: {dv_config_path}")
    
    diarization_temp_output_dir = "" # 화자 분리 시 생성되는 임시 오디오 세그먼트 저장 디렉토리
    speaker_segments = {} # 화자 분리 결과를 담을 딕셔너리
    
    try:
        # audio_path는 보통 /app/uploads/filename.wav 와 같은 형태일 것으로 가정
        # 해당 파일의 디렉토리 내에 temp_diarization_segments 폴더 생성
        diarization_temp_output_dir = os.path.join(os.path.dirname(audio_path), "temp_diarization_segments")
        os.makedirs(diarization_temp_output_dir, exist_ok=True)
        
        # diarization_utils.py의 split_speakers 함수 호출
        speaker_segments = split_speakers(audio_path, out_dir=diarization_temp_output_dir)
        logger.info(f"Diarization result for {audio_path}: {speaker_segments}")
    except Exception as e:
        logger.error(f"Diarization process critically failed for {audio_path}: {e}", exc_info=True)
        # 화자 분리 전체가 실패하면, 원본 오디오를 단일 화자("speaker_me")로 간주하고 처리
        speaker_segments = {"speaker_me": audio_path} 

    analysis_results = {}
    segment_files_to_delete_finally = []

    # speaker_segments가 비어있거나 딕셔너리가 아닌 경우에 대한 예외 처리
    if not speaker_segments or not isinstance(speaker_segments, dict):
        logger.error(f"No speaker segments returned from diarization for {audio_path}. Cannot proceed with analysis.")
        return {"error": "화자 분리 실패로 오디오 분석을 진행할 수 없습니다."}


    # 분리된 각 화자 세그먼트에 대해 분석 수행
    for speaker_id, segment_audio_path in speaker_segments.items():
        # segment_audio_path 유효성 검사
        if not segment_audio_path or not os.path.exists(segment_audio_path):
            logger.error(f"Invalid or non-existent segment path for speaker {speaker_id}: '{segment_audio_path}'. Skipping this segment.")
            analysis_results[speaker_id] = {"error": "세그먼트 파일 오류", "final_decision": "분석 불가"}
            continue

        logger.info(f"Processing segment for {speaker_id} from path: {segment_audio_path}")
        
        # 원본 오디오 파일이 아닌, 화자 분리로 생성된 임시 파일만 삭제 대상에 추가
        if segment_audio_path != audio_path and diarization_temp_output_dir and segment_audio_path.startswith(diarization_temp_output_dir):
            segment_files_to_delete_finally.append(segment_audio_path)

        # STT (음성 텍스트 변환)
        transcribed_text = "(STT 실패)"
        try:
            transcribed_text = transcribe_segment(segment_audio_path)
            logger.info(f"STT result for {speaker_id} (first 50 chars): {transcribed_text[:50]}...")
        except Exception as e_stt:
            logger.error(f"STT failed for segment {segment_audio_path} of speaker {speaker_id}: {e_stt}", exc_info=True)

        # KoBERT 텍스트 분석
        kobert_analysis_result = {"phishing_detected": False, "llm_score": 0.0, "final_label": "아님"}
        try:
            if transcribed_text != "(STT 실패)" and transcribed_text.strip():
                kobert_analysis_result = ensemble_inference(transcribed_text)
                logger.info(f"KoBERT analysis result for {speaker_id}: {kobert_analysis_result}")
            else:
                logger.warning(f"Skipping KoBERT analysis for {speaker_id} due to STT failure or empty text.")
        except Exception as e_kobert:
            logger.error(f"KoBERT analysis failed for text from speaker {speaker_id}: {e_kobert}", exc_info=True)

        # 딥보이스 탐지
        deepfake_probability = 0.0
        try:
            logger.debug(f"Predicting deepvoice for {segment_audio_path} with model {dv_model_path}")
            deepfake_probability = deepvoice_predict(segment_audio_path, dv_model_path, dv_config_path) 
            if deepfake_probability is None: # predict_deepvoice가 None을 반환할 수 있으므로 처리
                deepfake_probability = 0.0 
            logger.info(f"Deepvoice score (probability) for {speaker_id}: {deepfake_probability:.4f}")
        except Exception as e_dv:
            logger.error(f"Deepvoice prediction failed for segment {segment_audio_path} of speaker {speaker_id}: {e_dv}", exc_info=True)

        # 결과 취합 로직
        is_text_phishing = bool(kobert_analysis_result.get("phishing_detected", False))
        deepfake_detection_threshold = 0.5 
        is_voice_deepfake = deepfake_probability > deepfake_detection_threshold

        is_overall_phishing = is_text_phishing or is_voice_deepfake
        final_decision_message = "보이스피싱입니다!" if is_overall_phishing else "정상입니다."

        analysis_results[speaker_id] = {
            "text": transcribed_text,
            "phishing_detected_text": is_text_phishing,
            "text_score": kobert_analysis_result.get("llm_score", 0.0), # ensemble_utils의 llm_score 사용
            "deepfake_score": round(deepfake_probability, 4), # 딥보이스 확률 (0~1)
            "deepfake_detected_voice": is_voice_deepfake, # 딥보이스 탐지 여부 (True/False)
            "phishing": is_overall_phishing, # 최종 보이스피싱 여부 플래그
            "final_decision": final_decision_message # 최종 판단 메시지
        }
    
    # 사용된 임시 화자 분리 세그먼트 파일들 삭제
    for path_to_delete in segment_files_to_delete_finally:
        if os.path.exists(path_to_delete):
            try:
                os.remove(path_to_delete)
                logger.debug(f"Successfully removed temporary segment file: {path_to_delete}")
            except OSError as e_remove:
                logger.error(f"Error removing temporary segment file {path_to_delete}: {e_remove}", exc_info=True)
    
    # 임시 화자 분리 디렉토리 삭제 (비어있을 경우)
    if diarization_temp_output_dir and os.path.exists(diarization_temp_output_dir):
        try:
            if not os.listdir(diarization_temp_output_dir): # 디렉토리가 비어있으면
                os.rmdir(diarization_temp_output_dir)
                logger.debug(f"Successfully removed empty temporary directory: {diarization_temp_output_dir}")
        except OSError as e_rmdir:
            logger.warning(f"Could not remove temporary directory {diarization_temp_output_dir} (it might not be empty or other issues): {e_rmdir}")
            
    logger.info(f"Audio analysis pipeline finished for: {audio_path}. Results: {analysis_results}")
    return analysis_results