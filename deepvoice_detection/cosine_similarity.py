# real_audio와 deepfake_audio 유사도 판별

import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
real_dir = BASE_DIR / "real_audio"
fake_dir = BASE_DIR / "deepfake_audio"

def extract_mfcc(audio_path, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

similarities = []
files_compared = 0

for fname in os.listdir(real_dir):
    real_path = os.path.join(real_dir, fname)
    fake_path = os.path.join(fake_dir, fname)
    if os.path.exists(fake_path):
        mfcc_real = extract_mfcc(real_path)
        mfcc_fake = extract_mfcc(fake_path)
        sim = cosine_similarity(mfcc_real, mfcc_fake)
        similarities.append(sim)
        files_compared += 1
        print(f"{fname}: 유사도={sim:.3f}")

if similarities:
    print(f"\n총 {files_compared}쌍의 평균 MFCC 유사도: {np.mean(similarities):.3f}")
else:
    print("비교할 쌍이 없습니다.")
