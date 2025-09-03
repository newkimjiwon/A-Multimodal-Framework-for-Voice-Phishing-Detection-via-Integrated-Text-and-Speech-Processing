# Filename: visualize_log.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
LOG_CSV_PATH = "../static/csv/kobert_train_log.csv"
OUTPUT_DIR = "./"
OUTPUT_FILENAME = "training_history.png"

def visualize_training_log_en(log_path, save_path):
    """
    Reads a training log CSV file, visualizes the metrics in English,
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
    LEGEND_SIZE = 30 # [수정] 범례 폰트 크기 14 -> 16
    TICK_SIZE = 12

    # 2. Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # --- Plot Accuracy and F1 Scores ---
    ax.plot(df['Epoch'], df['Train Acc'], 'o-', label='Train Accuracy', color='blue')
    ax.plot(df['Epoch'], df['Validation Acc'], 's-', label='Validation Accuracy', color='cyan')
    ax.plot(df['Epoch'], df['Train F1'], 'o--', label='Train F1 Score', color='green')
    ax.plot(df['Epoch'], df['Validation F1'], 's--', label='Validation F1 Score', color='lime')
    
    # --- Set titles and labels ---
    ax.set_title('Model Accuracy & F1 Score per Epoch', fontsize=TITLE_SIZE)
    ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
    ax.set_ylabel('Score', fontsize=LABEL_SIZE)
    
    ax.set_xticks(df['Epoch'])
    ax.set_ylim(0.95, 1.005)
    
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
    visualize_training_log_en(LOG_CSV_PATH, save_file_path)