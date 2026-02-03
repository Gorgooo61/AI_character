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

LLM_models = {
    "base_model": "turboderp/Llama-3-8B-Instruct-exl2",
    "meta_model_inst": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta_model": "meta-llama/Meta-Llama-3-8B"
}

LLM_params = {
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
}

emotion_config = {
    "model_name": "j-hartmann/emotion-english-distilroberta-base",
    "device": "cuda",
    "max_length": 256,
}

tts_config = {
    "engine": "orpheus",    # "orpheus" | "kokoro" | "coqui" | "elevenlabs"

    # shared
    "output_device_index": None,    # default output
    "enabled": True,    # turn on/off tts engines

    # orpheus
    "orpheus": {
        "api_url": "http://127.0.0.1:1234", # LM Studio default
        "voice": "tara",
        "timeout_sec": 2.0,
    },

    # kokoro
    "kokoro": {
        "voice": "af_heart",
        "speed": 1.0,
        "debug": False,
    },

    # coqui
    "coqui": {
        "voice_reference": None,    # zero shot voice cloning (coudl be added later, format: wav, 6-12 sec)
        "speed": 1.1,
        "language": "hu",   # hu, en
        "device": "cuda",
        "use_deepspeed": False, # VRAM optimizer, should be insstalled separetly
    },

    # elevenlabs
    "elevenlabs": {
        "api_key": None,    # set later
        "voice_id": None,   # optional, engine may default
        "model_id": "eleven_multilingual_v2",
        "output_format": "wav_24000", # options: wav_16000, wav_22050, wav_24000, wav_32000, wav_44100, wav_48000, wav_8000
    },
}