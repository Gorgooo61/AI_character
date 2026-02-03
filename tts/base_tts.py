from __future__ import annotations

class BaseTTS:
    def __init__(self, signals, output_device_index=None):
        self.signals = signals
        self.output_device_index = output_device_index
        self.enabled = True

    def play(self, text: str, emotion_label = None):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError