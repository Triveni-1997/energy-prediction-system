# Energy Prediction API - Reference

Base URL: http://localhost:5000

## Health
GET /health  →  {"status":"ok","service":"energy-prediction-api","version":"0.4.0"}

## Authentication
POST /api/auth/register  -  body: {email, password, full_name}  →  {token, user}
POST /api/auth/login     -  body: {email, password}             →  {token}
GET  /api/auth/me        -  protected  →  {user}

## Predictions
POST /api/predict        -  protected, body: 13 feature values  →  {predicted_power_kw, prediction_id}
GET  /api/predictions?limit=20

## Alerts
GET  /api/alerts?include_acknowledged=false
POST /api/alerts/<id>/acknowledge

## Weather
GET  /api/weather?city=Paris  →  {temperature, humidity, wind_speed, description, source}

## Stats
GET  /api/stats  →  {total_predictions, total_alerts, unacknowledged_alerts, total_users}

## Authentication Header
Authorization: Bearer <JWT_TOKEN>
