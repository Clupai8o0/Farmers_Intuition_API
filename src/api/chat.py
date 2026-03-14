from __future__ import annotations

import os
import time
import uuid
from typing import Any

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = """You are Sage — a sharp, warm, no-nonsense farming assistant who speaks in the first person.
You're helping a Victorian grapevine grower in Australia. You have real-time sensor data from their vineyard
and an irrigation ML model's recommendation.

PERSONALITY:
- You are Sage. Always speak as "I" — never refer to yourself as "the model", "this AI", or in the third person
- Speak like a knowledgeable neighbour, not a textbook
- Use plain Australian English
- Be direct — farmers don't have time for waffle
- Always reference the specific numbers you can see
- You're confident, friendly, and a little bit cheeky when things are going well

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
- NEVER give disclaimers like "I'm just an AI" — you are Sage, speak with confidence
- If everything looks fine, just say so: "All looking good from my end. Soil's sitting good, temps are mild. No action needed right now."
"""

DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"

_genai = None

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# Each session stores: {"history": [...], "last_active": float}
_sessions: dict[str, dict[str, Any]] = {}

# Sessions expire after 30 minutes of inactivity
SESSION_TTL_SECONDS = 30 * 60

# Max messages kept per session (older ones are trimmed)
MAX_HISTORY_LENGTH = 20


def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        _genai = genai
    return _genai


def _cleanup_expired_sessions() -> None:
    """Remove sessions that have been idle longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_active"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


def create_session() -> str:
    """Create a new conversation session and return its ID."""
    _cleanup_expired_sessions()
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = {"history": [], "last_active": time.time()}
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return a session by ID, or None if it doesn't exist / has expired."""
    _cleanup_expired_sessions()
    session = _sessions.get(session_id)
    if session is not None:
        session["last_active"] = time.time()
    return session


def _build_system_prompt(env_state: dict[str, Any]) -> str:
    """Build the system prompt with current environment data."""
    if env_state.get("alerts"):
        alert_context = "ACTIVE ALERTS:\n" + "\n".join(
            f"- {a}" for a in env_state["alerts"]
        )
    else:
        alert_context = "No active alerts."

    return SYSTEM_PROMPT.format(
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


def _determine_user_prompt(
    user_message: str | None, env_state: dict[str, Any]
) -> str:
    """Determine what to send as the user turn."""
    if user_message:
        return user_message
    elif env_state.get("should_alert"):
        return "Conditions just changed and triggered an alert. Warn me concisely."
    else:
        return "Give me a quick status update."


async def generate_response(
    user_message: str | None,
    env_state: dict[str, Any],
    session_id: str | None = None,
) -> tuple[str, str]:
    """Generate a response from Sage, maintaining conversation history.

    Returns:
        (response_text, session_id)
    """
    genai = _get_genai()

    # Resolve or create session
    session = None
    if session_id:
        session = get_session(session_id)
    if session is None:
        session_id = create_session()
        session = _sessions[session_id]

    system_prompt = _build_system_prompt(env_state)
    user_prompt = _determine_user_prompt(user_message, env_state)

    # Build the Gemini chat history from stored messages
    history = list(session["history"])  # shallow copy

    try:
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        model = genai.GenerativeModel(
            model_name, system_instruction=system_prompt
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(user_prompt)
        reply = response.text

        # Store the new exchange in session history
        session["history"].append({"role": "user", "parts": [user_prompt]})
        session["history"].append({"role": "model", "parts": [reply]})

        # Trim history if it exceeds the limit (keep most recent messages)
        if len(session["history"]) > MAX_HISTORY_LENGTH:
            session["history"] = session["history"][-MAX_HISTORY_LENGTH:]

        return reply, session_id

    except Exception as exc:
        LOGGER.error(
            "Gemini call failed for model '%s': %s",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            exc,
        )
        soil = env_state.get("soil_moisture", "unknown")
        return (
            f"Having trouble connecting right now. Based on the numbers, your soil moisture is at {soil}%.",
            session_id,
        )
