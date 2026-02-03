import queue
import time


class Signals:
    def __init__(self, debug_print = True):
        self._debug_print = bool(debug_print)

        self._user_talking = False
        self._ai_talking = False
        self._ai_generating = False
        self._memory_generating = False
        self._new_q = False
        self._emotion_label = None

        self._last_event_time = 0.0

        # wweb signals
        self._stt_enabled = True
        self._avatar_enabled = True

        # simple event bus: (event_name, value, ts)
        self.sio_queue = queue.SimpleQueue()

    def _emit(self, name, value):
        ts = time.time()
        self._last_event_time = ts
        self.sio_queue.put((name, value, ts))
        if self._debug_print:
            print(f"[SIGNAL] {name} -> {value}")

    # last event timer
    @property
    def last_event_time(self):
        return self._last_event_time

    # user_talking
    @property
    def user_talking(self):
        return self._user_talking

    @user_talking.setter
    def user_talking(self, value):
        value = bool(value)
        if value == self._user_talking:
            return
        self._user_talking = value
        self._emit("user_talking", value)

    # ai_talking
    @property
    def ai_talking(self):
        return self._ai_talking

    @ai_talking.setter
    def ai_talking(self, value):
        value = bool(value)
        if value == self._ai_talking:
            return
        self._ai_talking = value
        self._emit("ai_talking", value)

    # ai_generating
    @property
    def ai_generating(self):
        return self._ai_generating

    @ai_generating.setter
    def ai_generating(self, value):
        value = bool(value)
        if value == self._ai_generating:
            return
        self._ai_generating = value
        self._emit("ai_generating", value)

    # memory_generating
    @property
    def memory_generating(self):
        return self._memory_generating

    @memory_generating.setter
    def memory_generating(self, value):
        value = bool(value)
        if value == self._memory_generating:
            return
        self._memory_generating = value
        self._emit("memory_generating", value)

    # new_queue input
    @property
    def new_q(self):
        return self._new_q

    @new_q.setter
    def new_q(self, value):
        value = bool(value)
        if value == self._new_q:
            return
        self._new_q = value
        self._emit("new_q", value)

    # emotion_label
    @property
    def emotion_label(self):
        return self._emotion_label

    @emotion_label.setter
    def emotion_label(self, value):
        value = None if value is None else str(value)
        if value == self._emotion_label:
            return
        self._emotion_label = value
        self._emit("emotion_label", value)

    # vtube studio hotkey
    @property
    def vts_last_hotkey(self):
        return self._vts_last_hotkey

    @vts_last_hotkey.setter
    def vts_last_hotkey(self, value):
        value = None if value is None else str(value)
        if value == getattr(self, "_vts_last_hotkey", None):
            return
        self._vts_last_hotkey = value
        self._emit("vts_last_hotkey", value)

    # stt switch for web
    @property
    def stt_enabled(self):
        return self._stt_enabled

    @stt_enabled.setter
    def stt_enabled(self, value):
        value = bool(value)
        if value == self._stt_enabled:
            return
        self._stt_enabled = value
        self._emit("stt_enabled", value)

    # tts+vts+emo switch for web
    @property
    def avatar_enabled(self):
        return self._avatar_enabled

    @avatar_enabled.setter
    def avatar_enabled(self, value):
        value = bool(value)
        if value == self._avatar_enabled:
            return
        self._avatar_enabled = value
        self._emit("avatar_enabled", value)