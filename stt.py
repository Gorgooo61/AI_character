import queue
import time
from RealtimeSTT import AudioToTextRecorder
from config import stt_config_realtime, stt_config_batch, stt_models
import logging

class SpeechRecognizer:
    def __init__(self, input_queue: queue.Queue):
        self.input_queue = input_queue
        self.active = False
        self.recorder = None

    def _on_text(self, text: str):
        """Callback for recognized text."""
        text = text.strip()
        if text:
            payload = {
                "timestamp": time.time(),
                "source": "microphone",
                "text": text
            }
            self.input_queue.put(payload)
            print(f"[STT] Recognized: {text}")

    def on_recording_start(self):
        print("[STT] Recording started")

    def on_recording_stop(self):
        print("[STT] Recording stopped")

    def start_realtime(self):
        config = {
            "spinner": False,
            "language": stt_config_realtime["language"],
            "use_microphone": True,
            "input_device_index": stt_config_realtime["input_device_index"],
            "silero_sensitivity": stt_config_realtime["silero_sensitivity"],
            "silero_use_onnx": True,
            "post_speech_silence_duration": 0.6,
            "min_length_of_recording": 0.5,
            "min_gap_between_recordings": 0.4,
            "enable_realtime_transcription": stt_config_realtime["enable_realtime_transcription"],
            "realtime_processing_pause": 0.4,
            "realtime_model_type": stt_config_realtime["realtime_model_type"],
            "use_main_model_for_realtime": True, # test config
            "compute_type": "auto",
            "device": stt_config_realtime["device"],
            "on_recording_start": self.on_recording_start,
            "on_recording_stop": self.on_recording_stop,
            "on_realtime_transcription_stabilized": self._on_text,
            "level": logging.ERROR

        }
        self.active = True

        with AudioToTextRecorder(**config) as recorder:
            self.recorder = recorder
            print("[STT] Ready and listening...")
            while self.active:
                recorder.text()  # Blocking until new transcription arrives

    def start_batch(self):
        config = {
            "spinner": False,
            "language": stt_config_batch["language"],
            "use_microphone": True,
            "input_device_index": stt_config_batch["input_device_index"],
            "silero_sensitivity": stt_config_batch["silero_sensitivity"],
            "silero_use_onnx": True,
            "post_speech_silence_duration": 0.4,
            "min_length_of_recording": 0,
            "min_gap_between_recordings": 0.2,
            "model": stt_models["large"],
            "compute_type": "auto",
            "device": stt_config_batch["device"],
            "on_recording_start": self.on_recording_start,
            "on_recording_stop": self.on_recording_stop,
            "level": logging.ERROR
        }

        self.active = True
        with AudioToTextRecorder(**config) as recorder:
            self.recorder = recorder
            print("[STT] Ready in batch mode (waiting for speech)...")

            while self.active:
                # Blocking call – waits until a full segment is finalized
                result = recorder.text()
                if result:
                    self._on_text(result)

    def stop(self):
        self.active = False
        if self.recorder:
            self.recorder.stop()
        print("[STT] Stopped")
