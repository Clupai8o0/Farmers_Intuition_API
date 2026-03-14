from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = """You are a concise voice assistant for a Victorian grapevine grower in Australia.
You have real-time sensor data from their vineyard and an irrigation ML model's recommendation.

PERSONALITY:
- Speak like a knowledgeable neighbour, not a textbook
- Use plain Australian English
- Be direct — farmers don't have time for waffle
- Always reference the specific numbers you can see

CURRENT VINEYARD STATE:
- Variety: {variety}
- Region: {region}
- Growth stage: {growth_stage}
- Temperature: {temperature}°C
- Humidity: {humidity}%
- Soil moisture: {soil_moisture}%
- Rainfall: {rainfall}mm
- Wind speed: {wind_speed}km/h

IRRIGATION MODEL OUTPUT:
- Predicted daily water need: {predicted_daily_l} litres
- Confidence: {confidence_level}
- Warnings: {warnings}
- Assumptions: {assumptions}

{alert_context}

RESPONSE RULES:
- For ALERTS: max 2 sentences. State what's wrong, state one action.
- For QUESTIONS: max 3 sentences. Answer directly, reference the numbers.
- For STATUS CHECKS: max 2 sentences. Say what looks good and what to watch.
- NEVER use technical jargon like "evapotranspiration" or "field capacity" — say "water use" and "how wet the soil is"
- NEVER give disclaimers like "I'm just an AI" — speak with confidence
- If everything looks fine, just say so: "All looking good. Soil's sitting at 62%, temps are mild. No action needed right now."
"""

_gemini_configured = False


def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        _gemini_configured = True


async def generate_response(user_message: str | None, env_state: dict[str, Any]) -> str:
    _ensure_gemini_configured()

    if env_state.get("alerts"):
        alert_context = "ACTIVE ALERTS:\n" + "\n".join(f"- {a}" for a in env_state["alerts"])
    else:
        alert_context = "No active alerts."

    prompt = SYSTEM_PROMPT.format(
        variety=env_state.get("variety", "unknown"),
        region=env_state.get("region", "unknown"),
        growth_stage=env_state.get("growth_stage", "unknown"),
        temperature=env_state.get("temperature", "N/A"),
        humidity=env_state.get("humidity", "N/A"),
        soil_moisture=env_state.get("soil_moisture", "N/A"),
        rainfall=env_state.get("rainfall", "N/A"),
        wind_speed=env_state.get("wind_speed", "N/A"),
        predicted_daily_l=env_state.get("predicted_daily_l", "N/A"),
        confidence_level=env_state.get("confidence_level", "N/A"),
        warnings=env_state.get("warnings", []),
        assumptions=env_state.get("assumptions", []),
        alert_context=alert_context,
    )

    if user_message:
        full_prompt = prompt + f'\n\nThe farmer just asked: "{user_message}"\nRespond concisely:'
    elif env_state.get("should_alert"):
        full_prompt = prompt + "\n\nConditions just changed and triggered an alert. Warn the farmer concisely:"
    else:
        full_prompt = prompt + "\n\nThe farmer wants a quick status update. Be brief:"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as exc:
        LOGGER.error("Gemini call failed: %s", exc)
        soil = env_state.get("soil_moisture", "unknown")
        return f"Having trouble connecting right now. Based on the numbers, your soil moisture is at {soil}%."
