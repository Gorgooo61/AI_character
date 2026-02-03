"""
NOTE: We intentionally do NOT use RealtimeTTS for ElevenLabs here.

RealtimeTTS includes an ElevenLabs wrapper (ElevenlabsEngine), but it relies on the
ElevenLabs "voices" catalog endpoint (https://api.elevenlabs.io/v1/voices) for
voice discovery/selection.

In that catalog, voices expose a list of verified languages (e.g., "verified_languages"
/ "labels.language"). In practice, the available entries we can select from do not
include Hungarian ("hu") / Hungarian locales (e.g., "hu-HU") in the verified language
set, which makes it hard to reliably choose a Hungarian-capable voice via the wrapper.

Because Hungarian support is one of the main reasons to use ElevenLabs in this project,
we integrate with the official ElevenLabs API directly. This gives us full control over:
- which ElevenLabs TTS model_id we call
- how we select or override voices (voice_id)
- language/locale handling and fallbacks
- streaming vs non-streaming response handling

(We can still keep RealtimeTTS for offline engines like Coqui/Kokoro/Orpheus, but for
ElevenLabs we prefer a direct integration path.)

API Reference: https://elevenlabs.io/docs/api-reference/text-to-speech
"""

import time
import os
import threading
from elevenlabs.client import ElevenLabs
from base_tts import BaseTTS
import wave
import pyaudio


class ElevenLabsTTS(BaseTTS):
    # emo tag need to be rewritten to elevenlabs logic
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
        output_device_index=None,
        *,
        api_key,
        voice_id,
        model_id = "eleven_multilingual_v2",
        output_format = "wav_24000",
        stability = 0.5,
        similarity_boost = 0.5,
        style = 0.0,
        use_speaker_boost = True,
        cache_dir = "cache/tts",
    ):
        super().__init__(signals, output_device_index=output_device_index)

        if not api_key:
            raise ValueError("ElevenLabsTTS: api_key is required.")
        if not voice_id:
            raise ValueError("ElevenLabsTTS: voice_id is required.")

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost

        if "wav" not in output_format: # always wav format
            self.output_format = "wav_24000"
        else:
            self.output_format = output_format
        self.file_extension = "wav"

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.client = ElevenLabs(api_key=api_key)

        # contorl like realtimetts (play_async, stop)
        self._play_thread = None
        self._stop_requested = False
        self._current_file_path = None
        self._pa_stream = None


    def generate_audio(self, text, file_name_no_ext = None):
        if file_name_no_ext is None:
            file_name_no_ext = f"tts_{int(time.time() * 1000)}"
        out_path = os.path.join(self.cache_dir, f"{file_name_no_ext}.{self.file_extension}")
        audio_iter = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings={
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost,
            },
        )
        with open(out_path, "wb") as f:
            for chunk in audio_iter:
                if self._stop_requested:
                    break
                if chunk:
                    f.write(chunk)
        if self._stop_requested:
            self.remove_file(out_path)
            return None           
        return out_path
    

    def remove_file(self, file_path: str):
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


    def play(self, text, emotion_label=None):
        """
        RealtimeTTS streams audio frames while synthesis is still ongoing (true low-latency streaming).
        With streaming engines (e.g. Coqui/Kokoro/Orpheus through RealtimeTTS), playback can start
        as soon as the first audio chunks are available.

        This ElevenLabs integration is simpler:
        1) We request a WAV file from ElevenLabs (network request).
        2) We write the full WAV to disk.
        3) We play it locally via PyAudio.
        """
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return
        tag = self._emotion_to_tag(emotion_label)
        if tag:
            text = f"{tag} {text}"

        # stop any current playback/generation first
        self.stop()

        self._stop_requested = False
        self._audio_started()

        def worker():
            pa = None
            stream = None
            wf = None
            path = None
            try:
                # generate wav
                path = self.generate_audio(text)
                self._current_file_path = path
                if not path:
                    return

                # play wav (pyaudio)
                wf = wave.open(path, "rb")
                pa = pyaudio.PyAudio()

                stream = pa.open(
                    format=pa.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.output_device_index,
                )
                self._pa_stream = stream

                frames_per_chunk = 1024
                data = wf.readframes(frames_per_chunk)

                while data:
                    if self._stop_requested:
                        break
                    stream.write(data)
                    data = wf.readframes(frames_per_chunk)

            except Exception as e:
                print(f"[ElevenLabsTTS] ERROR in play(): {e}")

            finally:
                try:
                    if wf is not None:
                        wf.close()
                except Exception:
                    pass
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                try:
                    if pa is not None:
                        pa.terminate()
                except Exception:
                    pass

                self._pa_stream = None
                self._current_file_path = None
                self._play_thread = None

                # remove cache file
                if path:
                    self.remove_file(path)

                self._audio_ended()

        self._play_thread = threading.Thread(target=worker, daemon=True)
        self._play_thread.start()

    def stop(self):
        self._stop_requested = True
        try:
            if self._pa_stream is not None:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
                self._pa_stream = None
        except Exception:
            pass
        self.signals.ai_talking = False
    
    def _emotion_to_tag(self, emotion_label):
        lab = (emotion_label or "").strip().lower()
        return self._EMO_TO_TAG.get(lab, "")

    def _audio_started(self):
        self.signals.ai_talking = True

    def _audio_ended(self):
        self.signals.ai_talking = False