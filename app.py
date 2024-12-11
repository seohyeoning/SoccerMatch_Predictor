from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# 모델 로드
model_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/result/soccer_model.pkl"
model = load(model_path)

# 팀 이름과 ID 데이터프레임
team_ids = {
    'Man United': 0, 'Fulham': 1, 'Ipswich': 2, 'Liverpool': 3, 'Arsenal': 4,
    'Wolves': 5, 'Everton': 6, 'Brighton': 7, 'Newcastle': 8, 'Southampton': 9,
    "Nott'm Forest": 10, 'Bournemouth': 11, 'West Ham': 12, 'Aston Villa': 13,
    'Brentford': 14, 'Crystal Palace': 15, 'Chelsea': 16, 'Man City': 17,
    'Leicester': 18, 'Tottenham': 19, 'Burnley': 20, 'Luton': 21,
    'Sheffield United': 22, 'Leeds': 23, 'Watford': 24, 'Norwich': 25,
    'West Brom': 26
}

@app.route('/')
def home():
    return render_template('index.html', team_ids=team_ids)


import json

# Accuracy 파일 로드
accuracy_path = "C:/Users/user/Desktop/psh/project/Soccer_predict/result/accuracy.json"
with open(accuracy_path, "r") as f:
    accuracy_data = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    home_team_id = int(request.form.get('home_team'))
    away_team_id = int(request.form.get('away_team'))

    # 예측
    input_data = np.array([[home_team_id, away_team_id]])
    prediction = model.predict(input_data)
    predicted_home_goals = int(round(prediction[0][0]))
    predicted_away_goals = int(round(prediction[0][1]))

    home_team_name = list(team_ids.keys())[home_team_id]
    away_team_name = list(team_ids.keys())[away_team_id]

    return render_template(
        'result.html',
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        predicted_home_goals=predicted_home_goals,
        predicted_away_goals=predicted_away_goals,
        overall_accuracy=accuracy_data["Overall Accuracy"]
    )



if __name__ == '__main__':
    app.run(debug=True)