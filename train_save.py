import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import dump

# 저장된 데이터 로드
train_data = np.load("C:/Users/user/Desktop/psh/project/Soccer_predict/data/train_data.npz")
test_data = np.load("C:/Users/user/Desktop/psh/project/Soccer_predict/data/test_data.npz")

X_train = train_data['X_train']
y_train = train_data['y_train']
X_test = test_data['X_test']
y_test = test_data['y_test']

# 모델 학습
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 모델 저장
model_save_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/result/soccer_model.pkl"
dump(model, model_save_path)
print(f"Model saved to {model_save_path}")

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 예측값을 반올림하여 정수화
y_pred_rounded = np.round(y_pred).astype(int)
y_test_rounded = np.round(y_test).astype(int)

# 결과 평가
mse = mean_squared_error(y_test, y_pred)
accuracy_home = np.mean(y_pred_rounded[:, 0] == y_test_rounded[:, 0])  # 홈 팀 득점 정확도
accuracy_away = np.mean(y_pred_rounded[:, 1] == y_test_rounded[:, 1])  # 원정 팀 득점 정확도
overall_accuracy = np.mean((y_pred_rounded == y_test_rounded).all(axis=1))  # 전체 정확도


# overall_accuracy 저장
accuracy_save_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/result/accuracy.json"
import json
accuracy_data = {
    "Overall Accuracy": overall_accuracy
}
with open(accuracy_save_path, "w") as f:
    json.dump(accuracy_data, f)
print(f"Accuracy saved to {accuracy_save_path}")


results = {
    "Mean Squared Error": mse,
    "Home Accuracy": accuracy_home,
    "Away Accuracy": accuracy_away,
    "Overall Accuracy": overall_accuracy,
    "Predictions": y_pred[:5],  # 예측값 샘플
    "True Values": y_test[:5],  # 실제값 샘플
}
# DataFrame 생성
results_df = pd.DataFrame({
    "HomeTeamID": X_test[:5, 0],
    "AwayTeamID": X_test[:5, 1],
    "Predicted_Home_Goals": y_pred_rounded[:5, 0],
    "Predicted_Away_Goals": y_pred_rounded[:5, 1],
    "True_Home_Goals": y_test_rounded[:5, 0],
    "True_Away_Goals": y_test_rounded[:5, 1],
    "Match_Accuracy": (y_pred_rounded[:5] == y_test_rounded[:5]).all(axis=1)
})

# 정확도 및 MSE 출력
print(f"Mean Squared Error: {mse}")
print(f"Home Team Accuracy: {accuracy_home:.2f}")
print(f"Away Team Accuracy: {accuracy_away:.2f}")
print(f"Overall Match Accuracy: {overall_accuracy:.2f}")
print(results_df)
