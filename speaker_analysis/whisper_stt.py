import whisper
import logging
import os
import re # 특수문자 제거를 위해 추가 (이전 요구사항 반영)

# 이 모듈에 대한 로거 설정
# 실제 로깅 레벨 및 포맷은 server.py 같은 메인 애플리케이션에서
# logging.basicConfig를 통해 설정하는 것이 일반적입니다.
logger = logging.getLogger(__name__)

# --- Whisper 모델 로드 ---
# 사용할 모델명: "tiny", "base", "small", "medium", "large" 중에서 선택
# "medium" 모델은 "base"나 "small"보다 정확도가 높지만, 더 많은 메모리와 처리 시간을 요구합니다.
# CPU 환경에서는 "medium"도 상당한 시간이 소요될 수 있습니다.
MODEL_NAME = "small" # 사용자님이 "medium"으로 변경하신 것을 반영
model = None
model_load_error_message = None # 모델 로드 실패 시 에러 메시지 저장

def load_whisper_model_once():
    """
    Whisper 모델을 애플리케이션 시작 시 또는 첫 호출 시 한 번만 로드합니다.
    성공적으로 로드되면 True, 실패하면 False를 반환합니다.
    """
    global model, model_load_error_message
    if model is not None:
        logger.info(f"Whisper model '{MODEL_NAME}' is already loaded.")
        return True
    
    if model_load_error_message is not None: # 이전에 로드 실패한 경우
        logger.error(f"Skipping Whisper model load attempt due to previous error: {model_load_error_message}")
        return False

    try:
        logger.info(f"Attempting to load Whisper model: {MODEL_NAME}...")
        # device="cpu"를 명시하여 CPU에서 실행되도록 합니다.
        # Dockerfile에서 PyTorch CPU 버전을 설치했으므로 이 설정이 적합합니다.
        model = whisper.load_model(MODEL_NAME, device="cpu")
        logger.info(f"Whisper model '{MODEL_NAME}' loaded successfully onto CPU.")
        return True
    except Exception as e:
        model_load_error_message = str(e)
        logger.critical(f"CRITICAL: Failed to load Whisper model '{MODEL_NAME}': {e}", exc_info=True)
        # 모델 로드 실패는 심각한 문제이므로, STT 기능이 작동하지 않음을 명확히 합니다.
        # 애플리케이션 시작을 중단시키고 싶다면 여기서 raise RuntimeError를 사용할 수 있습니다.
        return False

# 모듈이 처음 임포트될 때 모델 로드를 시도합니다.
# 서버 환경에서는 보통 임포트 시점에 이 코드가 실행됩니다.
if not load_whisper_model_once():
    logger.warning(
        f"Initial Whisper model ('{MODEL_NAME}') load failed. "
        f"STT functionality will be unavailable unless manually reloaded or fixed. "
        f"Error: {model_load_error_message}"
    )

def clean_stt_text(text: str) -> str:
    """
    STT 결과 텍스트에서 불필요한 특수문자를 제거하고 기본적인 정리를 수행합니다.
    한국어, 영어 알파벳, 숫자, 그리고 기본적인 문장 부호(공백, .,?!,'")와 일부 괄호만 남깁니다.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 허용할 문자 패턴: 한글, 영어 알파벳 대소문자, 숫자, 공백, 마침표, 쉼표, 물음표, 느낌표, 작은따옴표, 큰따옴표
    # 추가적으로 괄호 () [] {} 정도는 허용할 수 있습니다.
    # 필요에 따라 이 정규표현식을 조절하세요.
    # 현재: 한글, 숫자, 공백만 허용
    cleaned_text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s]', '', text)
    
    # 여러 개의 연속된 공백을 하나의 공백으로 변경하고, 문자열 앞뒤의 공백 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def transcribe_segment(audio_path: str) -> str:
    """지정된 오디오 파일 경로에서 음성을 텍스트로 변환하고 정제합니다."""
    global model, model_load_error_message

    # 1. 모델 로드 상태 확인
    if model_load_error_message:
        logger.error(f"STT failed for '{audio_path}': Whisper model previously failed to load ({model_load_error_message}).")
        return "(STT 실패 - 모델 초기 로드 오류)"
    
    if model is None: # 모델이 로드되지 않은 경우 (로드 재시도)
        logger.warning(f"STT warning for '{audio_path}': Whisper model was None. Attempting to reload...")
        if not load_whisper_model_once() or model is None:
             logger.error(f"STT failed for '{audio_path}': Whisper model reload failed or still None.")
             return "(STT 실패 - 모델 재로드 실패)"
        logger.info(f"Whisper model '{MODEL_NAME}' reloaded successfully for STT on '{audio_path}'.")

    # 2. 오디오 파일 유효성 검사
    if not os.path.exists(audio_path):
        logger.error(f"STT failed: Audio file not found at '{audio_path}'")
        return "(STT 실패 - 파일 없음)"
    
    try:
        file_size = os.path.getsize(audio_path)
        if file_size < 1024: # 예: 1KB 미만 파일 (매우 짧은 오디오 또는 빈 파일 가능성)
            logger.warning(f"STT warning: Audio file at '{audio_path}' is very small ({file_size} bytes). Transcription may be empty or inaccurate.")
            # 너무 작은 파일은 Whisper 처리 전 미리 빈 문자열로 처리할 수도 있습니다.
            # 예를 들어, return ""
    except OSError as e_size:
        logger.error(f"STT failed: Could not get size of audio file at '{audio_path}': {e_size}")
        return "(STT 실패 - 파일 크기 확인 불가)"

    # 3. Whisper를 사용한 음성 텍스트 변환
    try:
        logger.info(f"Transcribing audio segment: {audio_path} with model '{MODEL_NAME}'")
        
        # language="ko" : 한국어 지정
        # fp16=False : CPU 환경에서는 이 옵션이 무시되거나 자동으로 False로 처리됩니다.
        #              GPU 환경에서는 혼합 정밀도(mixed precision) 사용 여부를 결정합니다.
        # beam_size, temperature 등의 추가 파라미터로 정확도/속도 조절 가능 (고급)
        transcription_result = model.transcribe(audio_path, language="ko", fp16=False) 
        
        raw_text = transcription_result["text"]
        logger.debug(f"Raw STT for '{audio_path}' (first 100 chars): {raw_text[:100]}...")
        
        # 4. 텍스트 정제 (특수문자 제거 등)
        cleaned_text = clean_stt_text(raw_text)
        logger.info(f"Cleaned STT for '{audio_path}' (first 100 chars): {cleaned_text[:100]}...")
        
        if not cleaned_text.strip() and raw_text.strip():
             logger.warning(f"STT warning: Cleaned text is empty for '{audio_path}', but raw text was not. Raw: '{raw_text[:50]}...'")
        
        return cleaned_text
    except Exception as e_transcribe:
        logger.error(f"STT error during transcription for '{audio_path}': {e_transcribe}", exc_info=True)
        return "(STT 실패)"

# 로컬에서 직접 이 파일을 실행하여 테스트할 때 사용 (선택 사항)
# if __name__ == '__main__':
#     # 테스트를 위한 로깅 기본 설정
#     logging.basicConfig(
#         level=logging.DEBUG, 
#         format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
#     )
#     
#     # 모델 로드 상태 확인
#     if model:
#         logger.info(f"Whisper STT model '{MODEL_NAME}' is loaded and ready for testing.")
#         
#         # 테스트할 오디오 파일 경로 (실제 파일 경로로 변경 필요)
#         # test_audio_file = "path/to/your/test_audio.wav" 
#         # if os.path.exists(test_audio_file):
#         #     transcribed_text_result = transcribe_segment(test_audio_file)
#         #     logger.info(f"--- Test Transcription Result for {test_audio_file} ---")
#         #     logger.info(transcribed_text_result)
#         # else:
#         #     logger.error(f"Test audio file not found: {test_audio_file}")
#     else:
#         logger.critical(f"Whisper STT model '{MODEL_NAME}' FAILED to load. Cannot run tests. Error: {model_load_error_message}")