import torch
from transformers import pipeline
from config import emotion_config


class EmotionDetector:
    def __init__(self):
        self.model_name = emotion_config["model_name"]
        self.max_length = emotion_config["max_length"]

        # device = 0  -> GPU:0
        # device = -1 -> CPU
        device_pref = (emotion_config["device"]).lower()
        if device_pref == "cuda" and torch.cuda.is_available():
            device = 0
        else:
            device = -1

        self._classifier = pipeline(
            task="text-classification",
            model=self.model_name,
            device=device,
            tokenizer_kwargs={
                "truncation": True,
                "max_length": self.max_length,
            },
        )

    def predict_label(self, text):
        text = (text or "").strip()
        if not text:
            return "neutral"
        result = self._classifier(text)
        return (result[0].get("label") or "neutral").lower()
