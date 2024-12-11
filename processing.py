# 데이터 출처: https://www.football-data.co.uk/englandm.php 

import pandas as pd
import numpy as np
import chardet

# 파일 경로 리스트
file_paths = [
    f"C:/Users/user/Desktop/psh/project/Soccer_predict/data/E0_{year}.csv"
    for year in range(2000, 2026)
]

data_frames = []
for file_path in file_paths:
    try:
        # 파일 인코딩 감지
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

        # 파일 읽기
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
        data_frames.append(df)
    except pd.errors.ParserError as e:
        print(f"Parser error in file {file_path}: {e}")
    except UnicodeDecodeError as e:
        print(f"Encoding error in file {file_path}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

# 데이터프레임 결합
if data_frames:
    data = pd.concat(data_frames, ignore_index=True)
    print("Data successfully concatenated.")
else:
    print("No valid dataframes were read.")
    exit()


# 필요한 열만 추출
relevant_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
filtered_data = data[relevant_columns]

# NaN 값이 있는 행 제거
filtered_data = filtered_data.dropna()
print("Rows with NaN values have been removed.")

# 팀 이름에 고유 인덱스 매핑
team_ids = {team: idx for idx, team in enumerate(pd.unique(filtered_data[['HomeTeam', 'AwayTeam']].values.ravel()))}
filtered_data['HomeTeamID'] = filtered_data['HomeTeam'].map(team_ids)
filtered_data['AwayTeamID'] = filtered_data['AwayTeam'].map(team_ids)

# 입력(X)과 출력(y) 분리
X = filtered_data[['HomeTeamID', 'AwayTeamID']].values
y = filtered_data[['FTHG', 'FTAG']].values


# 학습 데이터와 테스트 데이터 분리 (8:2 비율)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 학습 및 테스트 데이터를 각각 저장
train_npz_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/data/train_data.npz"
test_npz_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/data/test_data.npz"
np.savez(train_npz_path, X_train=X_train, y_train=y_train)
np.savez(test_npz_path, X_test=X_test, y_test=y_test)

print(f"Training data saved to {train_npz_path}")
print(f"Test data saved to {test_npz_path}")