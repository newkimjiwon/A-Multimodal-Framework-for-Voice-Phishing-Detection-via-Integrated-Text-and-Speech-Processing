# ML\speaker_analysis\test_speaker_analysis.py
import argparse
from speaker_pipeline import analyze_multi_speaker_audio
import json

def main():
    parser = argparse.ArgumentParser(description="화자 분리 + STT + 분석 테스트")
    parser.add_argument("--input", type=str, required=True, help="입력 오디오 파일 경로")
    args = parser.parse_args()

    result = analyze_multi_speaker_audio(args.input)
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
