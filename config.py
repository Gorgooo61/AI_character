stt_config_realtime = {
    "realtime_model_type": "small",       # Real-time model (tiny.en, small, medium, ...)
    "language": "hu",
    "input_device_index": None,           # Default mic (set an index if multiple mics exist)
    "silero_sensitivity": 0.6,            # VAD sensitivity
    "device": "cuda",
    "enable_realtime_transcription": True,# Stream text in real-time while speaking
}

stt_config_batch = {
    "model": "small",
    "language": "hu",
    "input_device_index": None,           # Default mic (set an index if multiple mics exist)
    "silero_sensitivity": 0.6,            # VAD sensitivity
    "device": "cuda",
    "enable_realtime_transcription": False,# Stream text in real-time while speaking
    
}

stt_models = {
    "tiny_fp16": "models/whisper-hu-tiny-finetuned-V2-fp16",
    "tiny_fp32": "models/whisper-hu-tiny-finetuned-V2-fp32",
    "base_fp16": "models/whisper-hu-base-ct2-fp16",
    "base_fp32": "models/whisper-hu-base-ct2-fp32",
    "turbo_fp16": "models/whisper-v3-turbo-hu-ct2-fp16"
}

stt_mode = {
    "mode": "batch"
}