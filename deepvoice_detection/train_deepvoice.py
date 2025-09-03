import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
import numpy as np
import os
import csv
import json
import argparse
import random
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AudioCNNLSTM(nn.Module):
    def __init__(self, model_config):
        super(AudioCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=model_config['conv_in_channels'],
            out_channels=model_config['conv_out_channels'],
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # MaxPool1d 이후 LSTM의 input_size는 conv1의 out_channels와 동일
        lstm_input_actual_size = model_config['conv_out_channels']
        self.lstm = nn.LSTM(
            input_size=lstm_input_actual_size,
            hidden_size=model_config['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(model_config['lstm_hidden_size'] * 2, model_config['fc1_out_features']) # bidirectional이므로 *2
        self.fc2 = nn.Linear(model_config['fc1_out_features'], model_config['num_classes'])
        # Softmax는 CrossEntropyLoss 사용 시 모델 출력에서 제거

    def forward(self, x):
        # x shape: (batch_size, n_mfcc, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x shape: (batch_size, conv_out_channels, new_sequence_length)
        # LSTM에 넣기 위해 permute: (batch_size, new_sequence_length, conv_out_channels)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] # 마지막 time-step의 hidden state 사용
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) # raw logits 반환
        return x

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_config):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_config = feature_config

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        features = extract_mfcc_features(audio_path, self.feature_config)
        if features is None: # 특징 추출 실패 시
            print(f"Warning: Could not extract features for {audio_path}. Returning None.")
            return None # DataLoader의 collate_fn에서 처리하도록 None 반환
        # Conv1D는 (batch, channels, length)를 기대하므로, MFCC (length, channels)에서 transpose 필요
        return torch.tensor(features, dtype=torch.float32).transpose(0, 1), torch.tensor(label, dtype=torch.long)

def extract_mfcc_features(audio_path, feature_config):
    n_mfcc = feature_config['n_mfcc']
    n_fft = feature_config['n_fft']
    hop_length = feature_config['hop_length']
    max_length = feature_config['max_length']
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T
        # mfccs shape: (time_steps, n_mfcc)
        if mfccs.shape[0] > max_length:
            mfccs = mfccs[:max_length, :]
        else:
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant', constant_values=(0,))
        return mfccs
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None: # collate_fn이 None을 반환한 경우 (배치 전체가 에러)
                continue
            inputs, targets = batch_data
            if inputs is None or targets is None: # 혹시 모를 내부 None 체크
                continue

            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    if not all_targets: # 처리된 데이터가 없는 경우
        return {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics

# DataLoader에서 None을 반환하는 __getitem__을 처리하기 위한 collate_fn
def collate_fn_skip_none(batch):
    # batch는 (features, label) 튜플의 리스트
    # features가 None인 샘플 제거
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None # 모든 샘플이 유효하지 않으면 None 반환
    return torch.utils.data.dataloader.default_collate(batch)

def train_model(config):
    set_seed(config['training_params']['random_seed'])

    data_paths_config = config['data_paths']
    feature_config = config['feature_params']
    model_config = config['model_params']
    training_config = config['training_params']
    output_config = config['output_paths']

    real_audio_dir = data_paths_config['real_audio_dir']
    deepfake_audio_dir = data_paths_config['deepfake_audio_dir']

    real_files = [os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    deepfake_files = [os.path.join(deepfake_audio_dir, f) for f in os.listdir(deepfake_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]

    if not real_files or not deepfake_files:
        print(f"Error: Audio files not found. Check paths and file extensions.")
        print(f"Real audio files found: {len(real_files)} in {real_audio_dir}")
        print(f"Deepfake audio files found: {len(deepfake_files)} in {deepfake_audio_dir}")
        return

    all_files = real_files + deepfake_files
    all_labels = [0] * len(real_files) + [1] * len(deepfake_files) # 0: real, 1: deepfake

    # 데이터셋 분리 (학습용, 검증용)
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels,
        test_size=training_config['validation_split'],
        random_state=training_config['random_seed'],
        stratify=all_labels # 클래스 비율 유지
    )

    train_dataset = AudioDataset(train_files, train_labels, feature_config)
    val_dataset = AudioDataset(val_files, val_labels, feature_config)

    train_dataloader = DataLoader(
        train_dataset, batch_size=training_config['batch_size'], shuffle=True,
        num_workers=training_config['num_workers'], collate_fn=collate_fn_skip_none
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=training_config['batch_size'], shuffle=False,
        num_workers=training_config['num_workers'], collate_fn=collate_fn_skip_none
    )

    model = AudioCNNLSTM(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', # 검증 손실(val_loss) 기준
        patience=training_config['lr_scheduler_patience'],
        factor=training_config['lr_scheduler_factor'],
        verbose=True
    )

    log_path = output_config['log_csv_path']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "날짜", "Epoch", "Train Loss", "Train Acc", "Train F1",
            "Val Loss", "Val Acc", "Val Precision", "Val Recall", "Val F1", "Learning Rate"
        ])

    print(f"Starting training with config: {json.dumps(config, indent=2)}")
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(training_config['num_epochs']):
        model.train()
        train_loss_epoch = 0
        train_all_preds = []
        train_all_targets = []
        
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']} [Train]", unit="batch")
        for batch_data in loop:
            if batch_data is None: # collate_fn에 의해 전체 배치가 스킵된 경우
                print("Warning: Skipping a training batch due to all samples being invalid.")
                continue
            inputs, targets = batch_data
            if inputs is None or targets is None: # 추가적인 안전장치
                print("Warning: Skipping a training batch due to None inputs/targets after collation.")
                continue
            
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_targets.extend(targets.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        if not train_all_targets: # 에포크 동안 처리된 학습 데이터가 없는 경우
            print(f"Epoch [{epoch+1}/{training_config['num_epochs']}] - No training data processed. Check data and errors.")
            avg_train_loss, train_acc, train_f1 = 0, 0, 0
        else:
            avg_train_loss = train_loss_epoch / len(train_dataloader) if len(train_dataloader) > 0 else 0
            train_acc = accuracy_score(train_all_targets, train_all_preds)
            train_f1 = f1_score(train_all_targets, train_all_preds, average='binary', zero_division=0)

        # 검증
        val_metrics = evaluate_model(model, val_dataloader, criterion)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{training_config['num_epochs']}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f} | "
              f"LR: {current_lr:.6f}")

        with open(log_path, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1,
                round(avg_train_loss, 4), round(train_acc, 4), round(train_f1, 4),
                round(val_metrics['loss'], 4), round(val_metrics['accuracy'], 4),
                round(val_metrics['precision'], 4), round(val_metrics['recall'], 4), round(val_metrics['f1'], 4),
                current_lr
            ])

        # 학습률 스케줄러 업데이트 (검증 손실 기준)
        scheduler.step(val_metrics['loss'])

        # 최고 성능 모델 저장 (Val F1 기준)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs(os.path.dirname(output_config['best_model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), output_config['best_model_save_path'])
            print(f"Best model saved with Val F1: {best_val_f1:.4f} at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 조기 종료
        if epochs_no_improve >= training_config['early_stopping_patience']:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement on Val F1.")
            break
            
    # 마지막 에포크 모델 저장
    os.makedirs(os.path.dirname(output_config['model_save_path']), exist_ok=True)
    torch.save(model.state_dict(), output_config['model_save_path'])
    print(f"Last epoch model saved to {output_config['model_save_path']}")
    print(f"Training finished. Best validation F1-score: {best_val_f1:.4f}")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # Ensure conv_in_channels matches n_mfcc from feature_params
    if config['model_params']['conv_in_channels'] != config['feature_params']['n_mfcc']:
        print(f"Warning: model_params.conv_in_channels ({config['model_params']['conv_in_channels']}) "
              f"does not match feature_params.n_mfcc ({config['feature_params']['n_mfcc']}). "
              f"Adjusting conv_in_channels to n_mfcc.")
        config['model_params']['conv_in_channels'] = config['feature_params']['n_mfcc']
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deepvoice detection model with validation and enhanced metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    config_data = load_config(args.config) # Renamed to avoid conflict with 'config' module if any
    train_model(config_data)