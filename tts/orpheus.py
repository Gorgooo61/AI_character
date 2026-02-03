from RealtimeTTS import TextToAudioStream, OrpheusEngine, OrpheusVoice
import requests
from .base_tts import BaseTTS

# Optimal LM Studio GPU offload: ?
# Orpheus decoder run with realtimetts script and hardcoded on cuda
# Orpheus token gen run on LM Studio
# The decoding is streaming based

class OrpheusTTS(BaseTTS):
    # classifier labels -> Orpheus emotive tags
    _EMO_TO_TAG = {
        "joy": "<chuckle>",
        "anger": "<groan>",
        "sadness": "<sigh>",
        "surprise": "<gasp>",
        "fear": "<gasp>",
        "disgust": "<cough>",
        "neutral": "",  # no tag
    }

    def __init__(
        self,
        signals,
        api_url,
        voice = "tara",
        output_device_index=None,
        timeout_sec = 2.0,
    ):
        super().__init__(signals, output_device_index=output_device_index)
        self.api_url = (api_url or "").rstrip("/")
        completions_url = self.base_url + "/v1/completions"
        self.voice = (voice or "tara").strip().lower()
        self.timeout_sec = float(timeout_sec)

        self.engine = OrpheusEngine(api_url=completions_url)
        self.engine.set_voice(OrpheusVoice(self.voice))

        self.stream = TextToAudioStream(
            self.engine,
            output_device_index=self.output_device_index,
            on_audio_stream_start=self._audio_started,
            on_audio_stream_stop=self._audio_ended,
        )

    def check_connection(self): # LM studio
        url = self.api_url.rstrip("/") + "/api/v1/models"
        try:
            r = requests.get(url, timeout=self.timeout_sec)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # def set_voice(self, voice): # on the fly voice change
    #     voice = (voice or "").strip().lower()
    #     self.voice = voice
    #     self.engine.set_voice(OrpheusVoice(self.voice))

    def play(self, text, emotion_label = None):
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return

        # Map emotion label -> Orpheus tag
        tag = self._emotion_to_tag(emotion_label)
        if tag:
            text = f"{tag} {text}"

        # preffered realtimetts workflow
        def gen():
            yield text

        self.stream.feed(gen())
        self.stream.play_async(log_synthesized_text=False)

    def stop(self): # automatic interrupt logic not implemented yet
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.signals.ai_talking = False
        except Exception:
            pass

    def _emotion_to_tag(self, emotion_label):
        lab = (emotion_label or "").strip().lower()
        return self._EMO_TO_TAG.get(lab, "")

    def _audio_started(self):
        self.signals.ai_talking = True

    def _audio_ended(self):
        self.signals.ai_talking = False