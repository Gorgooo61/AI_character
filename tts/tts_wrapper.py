from __future__ import annotations

from config import tts_config

from .orpheus import OrpheusTTS
from .kokoro import KokoroTTS
from .coqui import CoquiTTS
from .elevenlabs import ElevenLabsTTS


def build_tts(signals):
    engine = (tts_config.get("engine") or "").strip().lower()
    output_device_index = tts_config.get("output_device_index", None)
    enabled = bool(tts_config.get("enabled", True))

    if engine == "orpheus":
        cfg = tts_config.get("orpheus", {}) or {}
        tts = OrpheusTTS(
            signals=signals,
            api_url=cfg.get("api_url", "http://127.0.0.1:1234"),
            voice=cfg.get("voice", "tara"),
            timeout_sec=cfg.get("timeout_sec", 2.0),
            output_device_index=output_device_index,
        )
        tts.enabled = enabled
        return tts

    if engine == "kokoro":
        cfg = tts_config.get("kokoro", {}) or {}
        tts = KokoroTTS(
            signals=signals,
            voice=cfg.get("voice", "af_heart"),
            speed=cfg.get("speed", 1.0),
            debug=cfg.get("debug", False),
            output_device_index=output_device_index,
        )
        tts.enabled = enabled
        return tts

    if engine == "coqui":
        cfg = tts_config.get("coqui", {}) or {}
        tts = CoquiTTS(
            signals=signals,
            voice_reference=cfg.get("voice_reference", None),
            language=cfg.get("language", "hu"),
            device=cfg.get("device", "cuda"),
            speed=cfg.get("speed", 1.1),
            use_deepspeed=cfg.get("use_deepspeed", False),
            output_device_index=output_device_index,
        )
        tts.enabled = enabled
        return tts

    if engine == "elevenlabs":
        cfg = tts_config.get("elevenlabs", {}) or {}
        tts = ElevenLabsTTS(
            signals=signals,
            api_key=cfg.get("api_key", None),
            voice_id=cfg.get("voice_id", None),
            model_id=cfg.get("model_id", "eleven_multilingual_v2"),
            output_format=cfg.get("output_format", "wav_24000"),
            output_device_index=output_device_index,
        )
        tts.enabled = enabled
        return tts

    raise ValueError(f"Unknown tts_config['engine']: {engine}")