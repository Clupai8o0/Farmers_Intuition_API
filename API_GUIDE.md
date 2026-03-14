# Farmers Intuition API Guide

Base URL: `https://farmers-intuition-api.vercel.app` (or `http://localhost:8000` locally)

---

## Endpoints Overview

| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| GET    | `/health`      | Health check & model status              |
| POST   | `/environment` | Push sensor data & get recommendations   |
| GET    | `/environment` | Get current environment state            |
| POST   | `/chat`        | Talk to Sage (AI assistant) with memory  |
| POST   | `/predict`     | Raw irrigation prediction                |
| POST   | `/recommend`   | Full irrigation recommendation           |
| POST   | `/retrain`     | Retrain the ML model                     |

---

## Core Workflow

The typical flow for a frontend is:

1. **Push sensor data** → `POST /environment`
2. **Chat with Sage** → `POST /chat` (uses the environment data automatically)
3. **Continue the conversation** → `POST /chat` with the returned `session_id`

---

## POST `/environment`

Push live sensor readings. The API runs the ML model and returns irrigation recommendations + alerts.

### Request

```json
{
  "temperature": 28.5,
  "humidity": 62.0,
  "soil_moisture": 45.0,
  "rainfall": 0.0,
  "wind_speed": 12.0,
  "growth_stage": "veraison",
  "variety": "shiraz",
  "region": "yarra_valley"
}
```

### Response

```json
{
  "status": "ok",
  "recommendation": {
    "baseline_weekly_l": 1200.0,
    "recommended_weekly_l": 980.0,
    "recommended_daily_l": 140.0,
    "confidence_level": "high",
    "assumptions": ["..."],
    "warnings": [],
    "feature_availability_summary": { "...": true },
    "model_name": "RandomForest"
  },
  "should_alert": false,
  "alerts": [],
  "environment": { "...full state..." }
}
```

### Alert Triggers

Sage and the API automatically detect:
- Soil moisture < 30% (critically low)
- Temperature > 35°C (heat stress)
- High humidity + warm temp + rainfall (downy mildew risk)
- Soil moisture > 85% (waterlogging)
- Irrigation need shifts > 20% between updates

---

## POST `/chat` — Talking to Sage

Sage is the AI voice assistant. She speaks in first person, uses plain Australian English, and references your live sensor data.

### Starting a New Conversation

```json
{
  "message": "How's my vineyard looking?"
}
```

### Response

```json
{
  "response": "All looking good from my end. Soil's sitting at 45%, temps are a comfortable 28 degrees. I'd keep the drip running at about 140 litres a day. No worries right now.",
  "session_id": "a1b2c3d4e5f6",
  "is_alert": false,
  "environment": { "...current state..." }
}
```

### Continuing a Conversation (Memory)

Send back the `session_id` from the previous response to keep the conversation going. Sage will remember what you've already discussed.

```json
{
  "message": "What about tomorrow if it hits 38 degrees?",
  "session_id": "a1b2c3d4e5f6"
}
```

Sage will remember the context of the previous exchange and respond accordingly.

### How Sessions Work

- **Omit `session_id`** → starts a fresh conversation
- **Include `session_id`** → continues from where you left off
- Sessions expire after **30 minutes** of inactivity
- Each session stores up to **20 messages** (oldest are trimmed)
- If an expired/invalid `session_id` is sent, a new session is created automatically

### Status Updates (No Message)

If you send an empty message, Sage gives a proactive status update:

```json
{
  "session_id": "a1b2c3d4e5f6"
}
```

If there are active alerts, she'll warn you instead.

---

## GET `/environment`

Returns the current environment state (last pushed sensor data + model output).

### Response (no data yet)

```json
{
  "status": "no_data",
  "message": "No sensor data received yet."
}
```

### Response (with data)

```json
{
  "status": "ok",
  "environment": {
    "temperature": 28.5,
    "humidity": 62.0,
    "soil_moisture": 45.0,
    "rainfall": 0.0,
    "wind_speed": 12.0,
    "growth_stage": "veraison",
    "variety": "shiraz",
    "region": "yarra_valley",
    "predicted_daily_l": 140.0,
    "predicted_weekly_l": 980.0,
    "confidence_level": "high",
    "warnings": [],
    "assumptions": ["..."],
    "should_alert": false,
    "alerts": [],
    "updated_at": "2026-03-15T10:30:00+00:00"
  }
}
```

---

## GET `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/irrigation_recommender.joblib"
}
```

---

## POST `/predict`

Raw ML prediction (without recommendations or adjustments).

```json
{
  "region": "Gippsland",
  "farm_id": "FARM_GIPPSLAND_001",
  "year": 2026,
  "quarter": "Q1",
  "week": 1,
  "nitrogen_weekly": 48.0,
  "phosphorus_weekly": 18.0,
  "potassium_weekly": 24.0,
  "calcium_weekly": 11.5,
  "magnesium_weekly": 7.8,
  "temperature_avg_c": 24.0,
  "sunlight_hours": 67.0,
  "humidity_percent": 56.0
}
```

---

## POST `/recommend`

Full recommendation with adjustments for growth stage, rainfall, etc.

Same fields as `/predict` plus optional:

```json
{
  "...same as predict...",
  "land_area_ha": 2.5,
  "crop_type": "shiraz",
  "growth_stage": "veraison",
  "rainfall_mm": 8.0,
  "soil_moisture_percent": 43.0
}
```

---

## Error Handling

All errors return standard HTTP status codes:

| Code | Meaning                                      |
|------|----------------------------------------------|
| 200  | Success                                      |
| 422  | Validation error (bad input)                 |
| 500  | Server error                                 |
| 503  | Model not loaded (run `/retrain` or deploy)  |

---

## Frontend Integration Example

```javascript
// 1. Push sensor data
const envRes = await fetch('/environment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    temperature: 28.5,
    humidity: 62,
    soil_moisture: 45,
    rainfall: 0,
    wind_speed: 12,
    growth_stage: 'veraison',
    variety: 'shiraz',
    region: 'yarra_valley'
  })
});

// 2. Start a chat with Sage
let sessionId = null;

const chatRes = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: "How's everything looking?" })
});
const chatData = await chatRes.json();
sessionId = chatData.session_id;  // Save this!

// 3. Continue the conversation
const followUp = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Should I irrigate today?",
    session_id: sessionId  // Pass it back
  })
});
```
