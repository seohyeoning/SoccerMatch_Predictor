from joblib import load
import numpy as np

# 저장된 모델 로드
model_save_path = "C:/Users/user/Desktop/psh/project/soccer_model.pkl"
model = load(model_save_path)

# 팀 ID로 예측 함수 정의
def predict_score(home_team_id, away_team_id):
    input_data = np.array([[home_team_id, away_team_id]])
    prediction = model.predict(input_data)
    predicted_home_goals = int(round(prediction[0][0]))
    predicted_away_goals = int(round(prediction[0][1]))
    return predicted_home_goals, predicted_away_goals

# 예제 입력
home_team_id = 0  # Man United
away_team_id = 1  # Fulham

predicted_home_goals, predicted_away_goals = predict_score(home_team_id, away_team_id)
print(f"Predicted score: HomeTeam {predicted_home_goals} - {predicted_away_goals} AwayTeam")
