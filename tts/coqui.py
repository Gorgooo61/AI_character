from typing import Optional
from RealtimeTTS import TextToAudioStream, CoquiEngine
from .base_tts import BaseTTS


class CoquiTTS(BaseTTS):
    def __init__(
        self,
        signals,
        voice_reference = None,
        language = "hu",
        device = "cuda",
        speed = 1.0,
        use_deepspeed = False,
        output_device_index=None,
    ):
        super().__init__(signals, output_device_index=output_device_index)

        # If you don't have reference wav, leave voice_reference=None.
        # CoquiEngine should fall back to a default voice configuration.
        kwargs = {
            "use_deepspeed": bool(use_deepspeed),
            "language": (language or "hu").strip(),
        }
        if device is not None:
            kwargs["device"] = (device or "").strip() or None
        if voice_reference:
            kwargs["voice"] = voice_reference
        if speed is not None:
            kwargs["speed"] = float(speed)

        self.engine = CoquiEngine(**kwargs)

        self.stream = TextToAudioStream(
            self.engine,
            output_device_index=self.output_device_index,
            on_audio_stream_start=self._audio_started,
            on_audio_stream_stop=self._audio_ended,
        )

    def play(self, text, emotion_label = None):
        # emotion_label ignored for Coqui (not supported)
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return

        self.stream.feed(text)
        self.stream.play_async(log_synthesized_text=False)

    def stop(self):
        try:
            self.stream.stop()
        except Exception:
            pass
        self.signals.ai_talking = False

    def _audio_started(self):
        self.signals.ai_talking = True

    def _audio_ended(self):
        self.signals.ai_talking = False
