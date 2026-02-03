import asyncio
import threading
import queue
import pyvts


class VTubeStudioController:
    DEFAULT_EMO_TO_HOTKEY = {
        "anger":   "exp_anger",
        "disgust": "exp_disgust",
        "fear":    "exp_fear",
        "joy":     "exp_joy",
        "neutral": "exp_neutral",
        "sadness": "exp_sadness",
        "surprise":"exp_surprise",
    }

    def __init__(
        self,
        signals,
        *,
        enabled = True,
        plugin_name = "AI VTS Plugin",
        developer = "Gorgooo61",
        token_path = "./vtubeStudio_token.txt",
        emo_to_hotkey = None,
    ):
        self.signals = signals
        self.enabled = bool(enabled)

        self._emo_to_hotkey = dict(self.DEFAULT_EMO_TO_HOTKEY)
        if emo_to_hotkey:
            for k, v in emo_to_hotkey.items():
                if k and v:
                    self._emo_to_hotkey[str(k).strip().lower()] = str(v).strip()

        self._q = queue.SimpleQueue()

        self._thread = None
        self._ready_evt = threading.Event()
        self._stop_evt = threading.Event()

        plugin_info = {
            "plugin_name": plugin_name,
            "developer": developer,
            "authentication_token_path": token_path,
        }
        self._vts = pyvts.vts(plugin_info=plugin_info)

# Public API
    def start(self):
        """Start background VTS loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background loop and close socket."""
        self._stop_evt.set()

    def trigger_emotion(self, emotion_label):
        """Trigger emotion hotkey based on label."""
        if not self.enabled:
            return
        emo = (emotion_label or "").strip().lower()
        hotkey = self._emo_to_hotkey.get(emo) or self._emo_to_hotkey.get("neutral")
        if hotkey:
            self._q.put(("trigger_hotkey", hotkey))

    def trigger_hotkey(self, hotkey_name):
        """Trigger a VTS hotkey by name."""
        if not self.enabled:
            return
        hotkey_name = (hotkey_name or "").strip()
        if not hotkey_name:
            return
        self._q.put(("trigger_hotkey", hotkey_name))

# Internal thread/async
    def _thread_entry(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        if not self.enabled:
            return

        # connect
        try:
            await self._vts.connect()
        except Exception as e:
            print(f"[VTubeStudio] ERROR: failed to connect: {e}")
            self.enabled = False
            return

        # auth token
        try:
            await self._vts.request_authenticate_token()
            await self._vts.request_authenticate()
        except Exception as e:
            print(f"[VTubeStudio] ERROR: auth failed: {e}")
            try:
                await self._vts.close()
            except Exception:
                pass
            self.enabled = False
            return

        self._ready_evt.set()
        print("[VTubeStudio] Connected & authenticated.")

        # main action loop
        while not self._stop_evt.is_set():
            if not self.enabled:
                await asyncio.sleep(0.1)
                continue

            # drain queue
            did_work  = 0
            while True:
                try:
                    action, data = self._q.get_nowait()
                except Exception:
                    break

                did_work  = True
                if action == "trigger_hotkey":
                    await self._trigger_hotkey_async(data)

            current_emo = (getattr(self.signals, "emotion_label", None) or "").strip().lower() or None
            if current_emo != self._last_seen_emotion:
                self._last_seen_emotion = current_emo
                if current_emo:
                    # emotion -> hotkey
                    hotkey = self._emo_to_hotkey.get(current_emo) or self._emo_to_hotkey.get("neutral")
                    if hotkey:
                        await self._trigger_hotkey_async(hotkey)
                        did_work = True

            # yield
            if not did_work:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0)

        # close
        try:
            await self._vts.close()
        except Exception:
            pass
        print("[VTubeStudio] Closed.")

    async def _trigger_hotkey_async(self, hotkey_name):
        if not hotkey_name:
            return
        try:
            req = self._vts.vts_request.requestTriggerHotKey(hotkey_name)
            resp = await self._vts.request(req)
            if resp and resp.get("messageType") == "APIError":
                msg = (resp.get("data") or {}).get("message", "unknown")
                print(f"[VTubeStudio] APIError on hotkey '{hotkey_name}': {msg}")
        except Exception as e:
            print(f"[VTubeStudio] trigger hotkey failed '{hotkey_name}': {e}")