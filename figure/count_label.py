import pandas as pd

# 파일 경로 설정
file_path = r'D:\Project\VishingProject\ML\static\csv\KoBERT_dataset_v3.0.csv'

# CSV 파일 로드
df = pd.read_csv(file_path)

# Label 0과 1의 개수 계산
counts = df['Label'].value_counts().rename_axis('Label').reset_index(name='Count')

# 결과 출력
print("Label 별 샘플 개수:")
print(counts)
