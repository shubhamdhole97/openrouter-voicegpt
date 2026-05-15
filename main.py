import os
import json
import base64
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
OPENROUTER_STT_MODEL = os.getenv("OPENROUTER_STT_MODEL", "openai/gpt-4o-mini-transcribe")
OPENROUTER_TTS_MODEL = os.getenv("OPENROUTER_TTS_MODEL", "openai/gpt-audio-mini")

APP_NAME = os.getenv("APP_NAME", "VoiceGPT Assistant")
SITE_URL = os.getenv("SITE_URL", "http://3.110.130.40:5173")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

app = FastAPI(title="VoiceGPT ESP32 OpenRouter Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://3.110.130.40:5173",
        SITE_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None


class TextChatResponse(BaseModel):
    answer_text: str
    model: str


def openrouter_headers(json_content=True):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
    }

    if json_content:
        headers["Content-Type"] = "application/json"

    return headers


def check_key():
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY missing in .env",
        )


@app.get("/")
def health():
    return {
        "status": "running",
        "app": APP_NAME,
        "chat_model": OPENROUTER_CHAT_MODEL,
        "stt_model": OPENROUTER_STT_MODEL,
        "tts_model": OPENROUTER_TTS_MODEL,
    }


def transcribe_audio_openrouter(wav_bytes: bytes) -> str:
    check_key()

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    payload = {
        "model": OPENROUTER_STT_MODEL,
        "input_audio": {
            "data": audio_b64,
            "format": "wav",
        },
        "language": "en",
    }

    try:
        r = requests.post(
            f"{OPENROUTER_BASE_URL}/audio/transcriptions",
            headers=openrouter_headers(),
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter STT failed: {exc}")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    text = data.get("text", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="No speech detected")

    return text


def ask_openrouter(question: str, system_prompt: Optional[str] = None) -> str:
    check_key()

    system_prompt = system_prompt or (
        "You are VoiceGPT Assistant. Give short and clear answers because "
        "the response will be spoken from a small ESP32 speaker."
    )

    payload = {
        "model": OPENROUTER_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
        "max_tokens": 250,
    }

    try:
        r = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=openrouter_headers(),
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter chat failed: {exc}")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid OpenRouter chat response: {data}",
        )


def tts_openrouter_pcm(answer_text: str) -> bytes:
    check_key()

    payload = {
        "model": OPENROUTER_TTS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a voice assistant. Speak the answer clearly.",
            },
            {
                "role": "user",
                "content": answer_text,
            },
        ],
        "modalities": ["text", "audio"],
        "audio": {
            "voice": "alloy",
            "format": "pcm16",
        },
        "stream": True,
    }

    try:
        r = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=openrouter_headers(),
            json=payload,
            timeout=120,
            stream=True,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter TTS failed: {exc}")

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    pcm_audio = bytearray()

    for line in r.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8", errors="ignore").strip()

        if not line.startswith("data: "):
            continue

        data_str = line.replace("data: ", "").strip()

        if data_str == "[DONE]":
            break

        try:
            data = json.loads(data_str)
        except Exception:
            continue

        try:
            choice = data["choices"][0]
            delta = choice.get("delta", {})

            audio_obj = (
                delta.get("audio")
                or delta.get("output_audio")
                or delta.get("audio_delta")
            )

            if isinstance(audio_obj, dict):
                audio_b64 = audio_obj.get("data") or audio_obj.get("audio")
                if audio_b64:
                    pcm_audio.extend(base64.b64decode(audio_b64))

            elif isinstance(audio_obj, str):
                pcm_audio.extend(base64.b64decode(audio_obj))

        except Exception:
            continue

    if len(pcm_audio) == 0:
        raise HTTPException(
            status_code=502,
            detail="No PCM audio received from OpenRouter TTS stream",
        )

    return bytes(pcm_audio)


@app.post("/api/text-chat", response_model=TextChatResponse)
def text_chat(req: TextChatRequest):
    question = req.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="message is required")

    answer = ask_openrouter(question, req.system_prompt)

    return TextChatResponse(
        answer_text=answer,
        model=OPENROUTER_CHAT_MODEL,
    )


@app.post("/api/esp32/voice-chat")
async def esp32_voice_chat(request: Request):
    wav_bytes = await request.body()

    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Audio body empty")

    print("ESP32 audio received:", len(wav_bytes), "bytes")

    question = transcribe_audio_openrouter(wav_bytes)
    print("Question:", question)

    answer = ask_openrouter(question)
    print("Answer:", answer)

    pcm_audio = tts_openrouter_pcm(answer)
    print("PCM audio size:", len(pcm_audio), "bytes")

    # IMPORTANT:
    # Do not put question/answer in HTTP headers.
    # Non-English text can break headers with UnicodeEncodeError.
    return Response(
        content=pcm_audio,
        media_type="audio/pcm",
    )
