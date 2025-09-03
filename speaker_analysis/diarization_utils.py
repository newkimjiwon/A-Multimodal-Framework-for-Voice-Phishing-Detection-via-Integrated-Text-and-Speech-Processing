# \ML\speaker_analysis\diarization_utils.py
import os
from collections import Counter
from resemblyzer import preprocess_wav, VoiceEncoder, audio as resemblyzer_audio
from pydub import AudioSegment
import numpy as np
from sklearn.cluster import KMeans
# import logging # 로깅 모듈 제거

# logger = logging.getLogger(__name__) # 로거 설정 제거

def split_speakers(audio_path, out_dir="temp_segments", min_samples_threshold=15):
    os.makedirs(out_dir, exist_ok=True)
    base_audio_filename = os.path.splitext(os.path.basename(audio_path))[0]

    wav = None
    internal_sampling_rate = 16000

    try:
        wav = preprocess_wav(audio_path)
        internal_sampling_rate = resemblyzer_audio.sampling_rate
    except Exception as e:
        # print(f"[DIARIZATION ERROR] preprocess_wav failed for {audio_path}: {e}") # 필요시 print로 대체
        try:
            audio = AudioSegment.from_file(audio_path)
            single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
            audio.export(single_path, format="wav")
            return {"speaker_me": single_path}
        except Exception: # 추가 예외 처리
            return {"speaker_me": audio_path}

    encoder = VoiceEncoder()
    cont_embeds = None
    wav_splits = None
    try:
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

        is_cont_embeds_valid = False
        if cont_embeds is not None:
            if isinstance(cont_embeds, np.ndarray):
                if cont_embeds.ndim > 0 and cont_embeds.shape[0] >= 2:
                    is_cont_embeds_valid = True
            elif isinstance(cont_embeds, list):
                if len(cont_embeds) >= 2 and all(isinstance(emb, np.ndarray) and emb.size > 0 for emb in cont_embeds):
                    is_cont_embeds_valid = True
        
        if not is_cont_embeds_valid:
            audio = AudioSegment.from_wav(audio_path)
            single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
            audio.export(single_path, format="wav")
            return {"speaker_me": single_path}

        features = np.array(cont_embeds)
        if features.ndim == 1:
             if features.shape[0] > 0:
                 features = features.reshape(features.shape[0], -1) 
                 if features.shape[1] == 0 and features.shape[0] > 0 : 
                      features = features.reshape(-1,1)
             else: 
                 features = np.array([]).reshape(0,1)

        if features.shape[0] < 2:
            audio = AudioSegment.from_wav(audio_path)
            single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
            audio.export(single_path, format="wav")
            return {"speaker_me": single_path}

    except Exception as e_embed:
        # print(f"[DIARIZATION ERROR] Embedding/feature creation for {audio_path}: {e_embed}")
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}

    actual_n_clusters = min(2, features.shape[0])
    if actual_n_clusters < 2:
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}
            
    try:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init='auto').fit(features)
        labels = kmeans.labels_
    except Exception as e_kmeans:
        # print(f"[DIARIZATION ERROR] KMeans clustering for {audio_path}: {e_kmeans}")
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}

    counts = Counter(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or (actual_n_clusters == 2 and min(counts.values()) < min_samples_threshold):
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}

    if wav_splits is None or len(labels) != len(wav_splits):
        # print(f"[DIARIZATION ERROR] Mismatch labels/wav_splits for {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}
            
    cluster_start_frames = {label: float('inf') for label in unique_labels}
    for i, label in enumerate(labels):
        current_split_info = wav_splits[i]
        start_frame = None
        if isinstance(current_split_info, slice):
            start_frame = current_split_info.start
        elif isinstance(current_split_info, (tuple, list)) and len(current_split_info) >= 1:
            start_frame = current_split_info[0]
        
        if start_frame is not None and start_frame < cluster_start_frames[label]:
            cluster_start_frames[label] = start_frame
    
    valid_labels_for_start_time = [l_ for l_ in cluster_start_frames if cluster_start_frames[l_] != float('inf')]
    if len(valid_labels_for_start_time) < 2:
        audio = AudioSegment.from_wav(audio_path)
        single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        audio.export(single_path, format="wav")
        return {"speaker_me": single_path}

    other_label = min(cluster_start_frames, key=cluster_start_frames.get)
    me_label_candidates = [l_ for l_ in unique_labels if l_ != other_label]
    me_label = me_label_candidates[0] if me_label_candidates else other_label 

    speakers_audio = {}
    if me_label is not None: 
        speakers_audio[me_label] = AudioSegment.silent(duration=0)
    if other_label is not None and other_label != me_label : 
        speakers_audio[other_label] = AudioSegment.silent(duration=0)
    
    for i, label in enumerate(labels):
        current_split_info = wav_splits[i]
        start_frame, end_frame = None, None

        if isinstance(current_split_info, slice):
            start_frame, end_frame = current_split_info.start, current_split_info.stop
        elif isinstance(current_split_info, (tuple, list)) and len(current_split_info) == 2:
            start_frame, end_frame = current_split_info
        
        if start_frame is None or end_frame is None or start_frame >= end_frame:
            continue

        try:
            segment_data_np = wav[start_frame:end_frame]
            if segment_data_np.size == 0:
                continue
            
            segment_data_int16 = (segment_data_np * 32767).astype(np.int16)
            segment_pydub = AudioSegment(
                segment_data_int16.tobytes(), 
                frame_rate=internal_sampling_rate,
                sample_width=segment_data_int16.dtype.itemsize, 
                channels=1
            )
        except Exception: # 간단한 예외 처리
            continue

        if label == me_label and me_label in speakers_audio:
             speakers_audio[me_label] += segment_pydub
        elif label == other_label and other_label in speakers_audio :
             speakers_audio[other_label] += segment_pydub
                 
    result_paths = {}
    final_me_audio = speakers_audio.get(me_label)
    final_other_audio = speakers_audio.get(other_label) if me_label != other_label else None

    if final_me_audio and len(final_me_audio) > 0:
        me_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
        try:
            final_me_audio.export(me_path, format="wav")
            result_paths["speaker_me"] = me_path
        except Exception: # 간단한 예외 처리
            pass # 실패 시 경로 추가 안 함
            
    if final_other_audio and len(final_other_audio) > 0:
        other_path = os.path.join(out_dir, f"speaker_other_{base_audio_filename}.wav")
        try:
            final_other_audio.export(other_path, format="wav")
            result_paths["speaker_other"] = other_path
        except Exception: # 간단한 예외 처리
            pass # 실패 시 경로 추가 안 함

    if not result_paths:
        try:
            audio_fallback = AudioSegment.from_file(audio_path)
            single_path = os.path.join(out_dir, f"speaker_me_{base_audio_filename}.wav")
            audio_fallback.export(single_path, format="wav")
            return {"speaker_me": single_path}
        except Exception:
            return {"speaker_me": audio_path} 
            
    return result_paths