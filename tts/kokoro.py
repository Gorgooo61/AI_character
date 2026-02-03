from typing import Optional
from RealtimeTTS import TextToAudioStream, KokoroEngine
from .base_tts import BaseTTS


class KokoroTTS(BaseTTS):
    def __init__(
        self,
        signals,
        voice = "af_heart",
        speed = 1.0,
        debug = False,
        output_device_index=None,
    ):
        super().__init__(signals, output_device_index=output_device_index)
        self.engine = KokoroEngine(debug=bool(debug))
        self.voice = (voice or "af_heart").strip()
        self.speed = float(speed)

        self.engine.set_voice(self.voice)
        self.engine.set_speed(self.speed)

        self.stream = TextToAudioStream(
            self.engine,
            output_device_index=self.output_device_index,
            on_audio_stream_start=self._audio_started,
            on_audio_stream_stop=self._audio_ended,
        )

    # def set_voice(self, voice): # on the fly voice change
    #     voice = (voice or "").strip().lower()
    #     self.voice = voice
    #     self.engine.set_voice(self.voice)

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