# Filename: visualize_deepfake_log.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- [수정] Configuration: Deepfake 모델에 맞게 경로 및 파일명 변경 ---
LOG_CSV_PATH = "../static/csv/train_deepfake_metrics_log.csv"
OUTPUT_DIR = "./"
OUTPUT_FILENAME = "deepfake_training_history.png"
# -----------------------------------------------------------------

def visualize_deepfake_log(log_path, save_path):
    """
    Reads a deepfake model's training log, visualizes the metrics,
    and saves the plot to an image file.
    """
    # 1. Load the log file
    try:
        df = pd.read_csv(log_path)
        print(f"✅ Log file loaded successfully: {log_path}")
    except FileNotFoundError:
        print(f"❌ Error: Log file not found. Please check the path: {os.path.abspath(log_path)}")
        return

    # --- Font size configuration ---
    TITLE_SIZE = 20
    LABEL_SIZE = 16
    LEGEND_SIZE = 30
    TICK_SIZE = 12

    # 2. Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # --- [수정] Plot Accuracy and F1 Scores using new CSV column names ---
    ax.plot(df['Epoch'], df['Train Acc'], 'o-', label='Train Accuracy', color='red')
    ax.plot(df['Epoch'], df['Val Acc'], 's-', label='Validation Accuracy', color='orange')
    ax.plot(df['Epoch'], df['Train F1'], 'o--', label='Train F1 Score', color='purple')
    ax.plot(df['Epoch'], df['Val F1'], 's--', label='Validation F1 Score', color='magenta')
    # -------------------------------------------------------------------
    
    # --- Set titles and labels ---
    ax.set_title('Deepfake Detection Model - Accuracy & F1 Score', fontsize=TITLE_SIZE)
    ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
    ax.set_ylabel('Score', fontsize=LABEL_SIZE)
    
    ax.set_xticks(df['Epoch'])

    # --- [수정] Y축 범위를 데이터에 맞게 조정 (0.8 ~ 1.01) ---
    # Epoch 6에서 값이 0.9 아래로 떨어지므로, 해당 부분을 포함하도록 범위를 넓힙니다.
    ax.set_ylim(0.8, 1.01)
    # ------------------------------------------------------
    
    # --- Set tick font size ---
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # --- Set legend location ---
    ax.legend(fontsize=LEGEND_SIZE, loc='lower right')
    
    ax.grid(True, linestyle='--', alpha=0.6)

    # 3. Adjust layout and save the file
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Graph saved successfully: {os.path.abspath(save_path)}")
    
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    save_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    visualize_deepfake_log(LOG_CSV_PATH, save_file_path)