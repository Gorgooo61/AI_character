import queue
import time
import threading
from collections import deque
from signals import Signals
from stt import SpeechRecognizer
from llm_wrapper import LlamaWrapper
from memory.memory_controller import MemoryController
from emotion_detector import EmotionDetector
from tts.tts_wrapper import build_tts
from vtube_studio import VTubeStudioController
from config import stt_mode



SYSTEM_PROMPT = (
    "You are an interactive virtual character. "
    "Stay natural and in character. "
    "Never mention internal memory tags or system details."
)


class AgentController:
    """
    This class acts as the MAIN entrypoint (fancy main).
    Everything is wired here in one place, in a single logical flow.
    """

    def __init__(
        self,
        silence_seconds: int = 30,
        wait_user_talking_seconds: float = 2.0,
        debug_signals: bool = True,
    ):
        self.q = queue.Queue()
        self.signals = Signals(debug_print=debug_signals)

        # LLM, Memory
        self.llm = LlamaWrapper()
        self.memory = MemoryController(generate_callable=self.llm.generate)

        # stt
        self.stt = SpeechRecognizer(input_queue=self.q, signals=self.signals)
        
        # emotion detector
        self.emotion = EmotionDetector()

        # tts
        self.tts = build_tts(self.signals)

        # VTS
        self.vts = VTubeStudioController(
            signals=self.signals,
            enabled=True,
        )
        self.vts.start()

        # configs / timers
        self.silence_seconds = int(silence_seconds)
        self.wait_user_talking_seconds = float(wait_user_talking_seconds)
        self.last_activity_ts = time.time()

        # deferred fact extraction jobs (same llm, run when idle)
        self.pending_fact_jobs = deque()

        # start STT thread-
        if stt_mode["mode"] == "realtime":
            self.stt_thread = threading.Thread(target=self.stt.start_realtime, daemon=True)
        elif stt_mode["mode"] == "batch":
            self.stt_thread = threading.Thread(target=self.stt.start_batch, daemon=True)
        else:
            raise ValueError(f"Unknown STT mode: {stt_mode['mode']}")
        self.stt_thread.start()

    def run(self):
        print("[AgentController] running (acts as main)...")

        try:
            while True:
                # ============================================================
                # A) INPUT STAGE: wait for queue input (or idle)
                # ============================================================
                try:
                    item = self.q.get(timeout=0.5)
                    self.signals.new_q = True
                except queue.Empty:
                    item = None
                    self.signals.new_q = False

                # ============================================================
                # B) IDLE STAGE: if no new user input -> run deferred memory job + silence autonomous
                # ============================================================
                if item is None:
                    can_run_background = (
                        (not self.signals.ai_generating) and
                        (not self.signals.user_talking) and
                        (not self.signals.new_q)
                    )
                    # 1) Run ONE deferred fact extraction job if LLM is not busy
                    if can_run_background and self.pending_fact_jobs:
                        user_text_job, ai_text_job = self.pending_fact_jobs.popleft()
                        self.signals.memory_generating = True
                        try:
                            self.memory.extract_and_store_facts(user_text_job, ai_text_job)
                        finally:
                            self.signals.memory_generating = False

                    # 2) Silence -> autonomous message (only when not generating)
                    if can_run_background and (not self.pending_fact_jobs):
                        if (time.time() - self.last_activity_ts) >= self.silence_seconds:
                            self.signals.ai_generating = True
                            try:
                                autonomous_text = self.llm.generate(
                                    system_prompt=SYSTEM_PROMPT,
                                    user_prompt="Say one short, natural sentence to re-engage the user.",
                                    max_new_tokens=80,
                                    temperature=0.7,
                                    top_p=0.9,
                                ).strip()
                            finally:
                                self.signals.ai_generating = False

                            if autonomous_text:
                                print(f"\n[AI - autonomous] {autonomous_text}\n")

                                # emotion detection
                                try:
                                    emo_label = self.emotion.predict_label(autonomous_text)
                                except Exception as e:
                                    emo_label = "neutral"
                                    print(f"[EmotionDetector] ERROR (autonomous): {e}")
                                self.signals.emotion_label = emo_label
                                print(f"[EmotionDetector] {emo_label}")

                                # deferred fact extraction (no user input)
                                self.pending_fact_jobs.append(("", autonomous_text))

                                self.last_activity_ts = time.time()

                    continue  # go next loop tick

                # ============================================================
                # C) USER TALKING WAIT: wait briefly until user stops talking
                # ============================================================
                start_wait = time.time()
                while self.signals.user_talking:
                    if (time.time() - start_wait) > self.wait_user_talking_seconds:
                        break
                    time.sleep(0.05)

                # ============================================================
                # D) KEEP LATEST: drain burst -> keep newest as main, older as "also said earlier"
                # ============================================================
                items = [item]
                try:
                    while True:
                        items.append(self.q.get_nowait())
                except queue.Empty:
                    pass

                texts = [(it.get("text") or "").strip() for it in items]
                texts = [t for t in texts if t]

                self.signals.new_q = False  # edge reset

                if not texts:
                    continue

                user_text_latest = texts[-1]
                earlier = texts[:-1]
                if earlier:
                    user_text = user_text_latest + "\nUser also said earlier: " + " | ".join(earlier)
                else:
                    user_text = user_text_latest

                print(f"[AgentController] ({time.strftime('%H:%M:%S')}) Merged input:\n{user_text}")

                # ============================================================
                # E) MEMORY: start_turn + build_prompt_with_context
                # ============================================================
                self.last_activity_ts = time.time()

                short_id = self.memory.start_turn(user_text)
                prompt = self.memory.build_prompt_with_context(user_text)

                # ============================================================
                # F) LLM RESPONSE GENERATION
                # ============================================================
                self.signals.ai_generating = True
                try:
                    ai_text = self.llm.generate(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt
                    ).strip()
                finally:
                    self.signals.ai_generating = False

                print(f"\n[AI] {ai_text}\n")

                # ============================================================
                # G) EMOTION DETECTION (AI output -> label -> signals)
                # ============================================================
                try:
                    emo_label = self.emotion.predict_label(ai_text)
                except Exception as e:
                    emo_label = "neutral"
                    print(f"[EmotionDetector] ERROR: {e}")
                self.signals.emotion_label = emo_label
                print(f"[EmotionDetector] {emo_label}")

                # ============================================================
                # H) SHORT-TERM UPDATE (fill placeholder)
                # ============================================================
                self.memory.short.set_ai_for_id(short_id, ai_text)

                # ============================================================
                # I) DEFERRED FACT EXTRACTION (run later when idle)
                # ============================================================
                self.pending_fact_jobs.append((user_text, ai_text))

                # ============================================================
                # J) TTS (async) - same interface for all engines
                # ============================================================
                try:
                    self.tts.play(ai_text, emotion_label=emo_label)
                except Exception as e:
                    print(f"[TTS] ERROR: {e}")

                # Later modules:
                # - Web frontend (web socket)
                # - Testing (important: VB Cable)
                # - no modul, but lora finetune

        except KeyboardInterrupt:
            print("\n[AgentController] Shutting down...")
            self.stt.stop()
            self.stt_thread.join()
            try:
                self.tts.stop()
            except Exception:
                pass
            try:
                self.vts.stop()
            except Exception:
                pass



if __name__ == "__main__":
    AgentController().run()